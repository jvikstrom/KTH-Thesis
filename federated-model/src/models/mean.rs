use crate::core::model::Model;
use crate::losses::L2;

pub struct Mean {
    /// Current estimate of mean
    pub mean: Vec<f32>
}

impl Model<
        /*model=*/Mean, 
        /*data=*/Vec<Vec<f32>>, 
        /*loss=*/L2,
        /*gradient=*/Vec<f32>> for Mean {

    /// Combines two models to form a new one:
    /// Averages the two means
    fn combine(&self, other: &Mean) -> Result<Box<Self>, String> {
        if self.mean.len() != other.mean.len() {
            return Err(format!("Mean dimension mismatch, {} != {}",
                    self.mean.len(),
                    other.mean.len()));
        }

        let mut n_mean: Vec<f32> = vec![0.0; self.mean.len()];
        for i in 0..self.mean.len() {
            n_mean[i] = (self.mean[i] + other.mean[i]) / 2.0;
        }

        return Ok(Box::new(Mean {
            mean: n_mean
        }));
    }

    /// Applies a gradient to the model: applies it as a minimization
    fn apply(&mut self, gradient: &Vec<f32>, learning_rate: f32) -> Result<&Self, String> {
        if self.mean.len() != gradient.len() {
            return Err(format!("Gradient not matching weights, {} != {}",
                    gradient.len(),
                    self.mean.len()));

        }

        for i in 0..self.mean.len() {
            self.mean[i] = self.mean[i] - learning_rate * gradient[i];
        }

        return Ok(self);

    }

    /// Takes the gradient wrt some data.
    /// We have f = (x - y)^2 / 2
    /// df/dy = -(x - y)
    fn gradient(&self, data: &Vec<Vec<f32>>) -> Result<Vec<f32>, String> {
        let mut gradient = vec![0.0; self.mean.len()];

        // Calculate sum of all gradients
        for sample in data {

            // Check that sample has valid length
            if self.mean.len() != sample.len() {
                return Err(format!("Sample not matching weights, {} != {}",
                        sample.len(),
                        self.mean.len()));

            }

            // Sum..
            for i in 0..sample.len() {
                gradient[i] += - (sample[i] - self.mean[i]);
            }
        }

        // Average gradient of all samples
        for i in 0..self.mean.len() {
            gradient[i] /= data.len() as f32;
        }


        return Ok(gradient);
    }

    /// Calculate L2 loss of all samples
    fn eval(&self, data: &Vec<Vec<f32>>) -> Result<f32, String> {
        let mut err = 0.0;
        for sample in data {
            if self.mean.len() != sample.len() {
                return Err(format!("Sample not matching weights, {} != {}",
                        sample.len(),
                        self.mean.len()));

            }

            for i in 0..sample.len() {
                let diff = sample[i] - self.mean[i];
                err += diff * diff;
            }
        }

        return Ok(err / data.len() as f32)


    }
}


#[cfg(test)]
mod tests {
    use rand::distributions::{Normal, Distribution};
    use super::{Mean, Model};

    #[test]
    fn mean_train() {
        let data_samples: usize = 100;
        let dimension: usize = 2;

        let n = Normal::new(5.0, 10.0);

        let mut expected_model = vec![0.0; dimension];

        let mut data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..data_samples {
            let mut sample: Vec<f32> = Vec::new();
            for d in 0..dimension {
                sample.push(n.sample(&mut rand::thread_rng()) as f32);
                expected_model[d] += sample[d];

            }

            data.push(sample);
        }

        for d in 0..dimension {
            expected_model[d] /= data_samples as f32;
        }

        let mut model = Mean {
            mean: vec![0.0; dimension]
        };


        let gradient_iter = 60;
        let learning_rate = 0.1;

        for e in 0..gradient_iter {
            let gradient = model.gradient(&data).unwrap();
            model.apply(&gradient, learning_rate).unwrap();

            println!("{} - Error: {} - Model: {:?}", 
                e, 
                model.eval(&data).unwrap(), 
                model.mean);
        }


        println!("Expected model: {:?}", expected_model);

        for d in 0..dimension {
            assert!(f32::abs(model.mean[d] - expected_model[d]) < 0.1);
        }
    }

    #[test]
    fn mean_combine() {
        let dimension: usize = 2;

        let m1 = Mean {
            mean: vec![0.0; dimension]
        };

        let m2 = Mean {
            mean: vec![1.0; dimension]
        };

        let target = Mean {
            mean: vec![0.5; dimension]
        };

        let m3 = m1.combine(&m2).unwrap();

        for d in 0..dimension {
            assert!(f32::abs(m3.mean[d] - target.mean[d]) < 0.0001);
        }
    }
}
