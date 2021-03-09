/*extern crate protobuf_codegen_pure;
*/
/*protobuf_codegen_pure::run!{protobuf_codegen_pure::Args {
    out_dir: "src/protos",
    input: &["protos/peer.proto", "protos/hypercraft.proto"],
    includes: &["protos"],
    customize: protobuf_codegen_pure::Customize {
      ..Default::default()
    },
}}.expect("protoc");
*/
/*
protobuf_codegen_pure::Args::new()
    .out_dir("src/protos")
    .inputs(&["protos/a.proto", "protos/b.proto"])
    .include("protos")
    .run()
    .expect("protoc");
*/

use protoc_rust::Customize;

fn main() {
    protoc_rust::Codegen::new()
        .out_dir("src/protos")
        .inputs(&["protos/learning.proto", "protos/hypercraft.proto"])
        .include("protos")
        .run()
        .expect("protoc");
}
