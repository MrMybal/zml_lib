const std = @import("std");
const training_steps = 1000;

pub const Ddim = struct {
    steps: usize,
    stride: usize,
    alpha_product: [training_steps]f32,

    pub fn init(steps: usize) !Ddim {
        if (steps == 0 or steps >= training_steps) return error.InvalidStepCount;
        const start = @sqrt(@as(f32, 0.0015));
        const end = @sqrt(@as(f32, 0.0195));
        const interval = (end - start) / (training_steps - 1);
        var product: f32 = 1;
        var result: Ddim = .{ .steps = steps, .stride = training_steps / steps, .alpha_product = undefined };
        for (&result.alpha_product, 0..) |*alpha, i| {
            const sigma = start + interval * @as(f32, @floatFromInt(i));
            product *= 1 - sigma * sigma;
            alpha.* = product;
        }
        return result;
    }

    pub fn timestep(self: *const Ddim, index: usize) !usize {
        if (index >= self.steps) return error.InvalidStep;
        return (self.steps - index - 1) * self.stride + 1;
    }

    /// Applies one eta=0 step with the UNet's predicted epsilon and no sample clipping.
    pub fn step(self: *const Ddim, index: usize, sample: []f32, epsilon: []const f32) !void {
        if (sample.len != epsilon.len) return error.InvalidBufferSize;
        const t = try self.timestep(index);
        const current = self.alpha_product[t];
        const previous = self.alpha_product[t -| self.stride];
        const signal_ratio = @sqrt(previous / current);
        const noise_ratio = @sqrt(1 - previous) - signal_ratio * @sqrt(1 - current);
        for (sample, epsilon) |*value, noise| value.* = signal_ratio * value.* + noise_ratio * noise;
    }
};

test "four DDIM steps match the local Diffusers schedule" {
    const schedule = try Ddim.init(4);
    const expected_timesteps = [_]usize{ 751, 501, 251, 1 };
    const expected_samples = [_][4]f32{
        .{ -0.0879074484, -1.5848908424, 2.6539819241, 5.7862720490 },
        .{ -0.4414763451, -3.2119762897, 5.5290904045, 11.4256525040 },
        .{ -0.8057097197, -4.4140110016, 7.7119951248, 15.4740867615 },
        .{ -0.8095448017, -4.4157290459, 7.7170090675, 15.4760847092 },
    };
    var sample = [_]f32{ 0.125, -0.5, 0.75, 2.0 };
    for (expected_timesteps, expected_samples, 0..) |t, output, i| {
        try std.testing.expectEqual(t, try schedule.timestep(i));
        try schedule.step(i, &sample, &.{ 0.2, -0.1, 0.05, 0.6 });
        for (output, sample) |reference, actual| try std.testing.expectApproxEqAbs(reference, actual, 1e-5);
    }
}
