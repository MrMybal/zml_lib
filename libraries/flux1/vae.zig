const runtime = @import("zml_lib/runtime");

const Decoder = runtime.kl_decoder.Decoder(.{ .channels = 16, .scaling = 0.3611, .shift = 0.1159 });
pub const channels = Decoder.channels;
pub const factor = Decoder.factor;
pub const schema = Decoder.schema;
pub const decode = Decoder.decode;
