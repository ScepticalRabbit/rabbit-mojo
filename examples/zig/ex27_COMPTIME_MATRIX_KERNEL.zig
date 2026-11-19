//! Fixed row-major matrix products; same runtime values as Python example 27.
const std = @import("std");
fn multiply(comptime N: usize, a: [N * N]f64, b: [N * N]f64) [N * N]f64 {
    var dst = [_]f64{0} ** (N * N);
    inline for (0..N) |i| {
        inline for (0..N) |j| {
            inline for (0..N) |k| dst[i * N + j] += a[i * N + k] * b[k * N + j];
        }
    }
    return dst;
}
test "separately generated 2x2 and 3x3 products" {
    inline for (.{ 2, 3 }) |N| {
        var a: [N * N]f64 = undefined;
        var b = [_]f64{0} ** (N * N);
        for (0..N * N) |i| a[i] = @floatFromInt(i + 1);
        for (0..N) |i| b[i * N + i] = 2;
        const result = multiply(N, a, b);
        for (0..N * N) |i| try std.testing.expectApproxEqAbs(a[i] * 2, result[i], 1e-13);
        std.debug.print("N={d} product={any}\n", .{ N, result });
    }
}
