
pub trait Model<M, Data, Loss, Gradient> {
    fn combine(&self, other: &M) -> Result<Box<Self>, String>;
    fn apply(&mut self, gradient: &Gradient, learning_rate: f32) -> Result<&Self, String>;
    fn gradient(&self, data: &Data) -> Result<Gradient, String>;
    fn eval(&self, data: &Data) -> Result<f32, String>;
}
