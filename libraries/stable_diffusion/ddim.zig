//! Deterministic DDIM with the SD v1.5 scaled-linear beta schedule.
const training_steps = 1000;

pub const Schedule = struct {
    steps: usize,
    stride: usize,
    alpha: [training_steps]f32,

    pub fn init(steps: usize) !Schedule {
        if (steps == 0 or steps > 100) return error.InvalidDDIMSteps;
        const low = @sqrt(@as(f32, 0.00085));
        const high = @sqrt(@as(f32, 0.012));
        const interval = (high - low) / (training_steps - 1);
        var result: Schedule = .{ .steps = steps, .stride = training_steps / steps, .alpha = undefined };
        var product: f64 = 1;
        for (&result.alpha, 0..) |*value, i| {
            const index: f32 = @floatFromInt(i);
            const root = low + interval * index;
            const alpha: f32 = 1 - root * root;
            product *= alpha;
            value.* = @floatCast(product);
        }
        return result;
    }

    pub fn timestep(self: *const Schedule, index: usize) usize {
        return (self.steps - index - 1) * self.stride + 1;
    }

    pub fn addNoise(self: *const Schedule, index: usize, source: []const f32, noise: []f32) void {
        const current = self.alpha[self.timestep(index)];
        const signal = @sqrt(current);
        const sigma = @sqrt(1 - current);
        for (noise, source) |*value, pixel| value.* = signal * pixel + sigma * value.*;
    }

    pub fn step(self: *const Schedule, index: usize, sample: []f32, epsilon: []const f32) void {
        const t = self.timestep(index);
        const current = self.alpha[t];
        const previous = self.alpha[t -| self.stride];
        const signal = @sqrt(current);
        const noise = @sqrt(1 - current);
        for (sample, epsilon) |*pixel, prediction| {
            const original = (pixel.* - noise * prediction) / signal;
            pixel.* = @sqrt(previous) * original + @sqrt(1 - previous) * prediction;
        }
    }
};
