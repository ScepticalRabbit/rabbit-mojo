//! Vec(T,N) owns a fixed array just like Mojo Vec[dtype,N].
const std = @import("std");
fn Vec(comptime T: type, comptime N: usize) type {
    return struct {
        data: [N]T,
        const Self = @This();
        fn dot(a: Self, b: Self) T {
            var value: T = 0;
            inline for (0..N) |i| value += a.data[i] * b.data[i];
            return value;
        }
    };
}
test "two precisions, three compile-time lengths" {
    inline for (.{ f32, f64 }) |T| {
        inline for (.{ 2, 3, 4 }) |N| {
            var a: Vec(T, N) = undefined;
            var b: Vec(T, N) = undefined;
            for (0..N) |i| {
                a.data[i] = @floatFromInt(i + 1);
                b.data[i] = @floatFromInt(N - i);
            }
            const expected: T = if (N == 2) 4 else if (N == 3) 10 else 20;
            try std.testing.expectApproxEqAbs(expected, a.dot(b), 1e-6);
            std.debug.print("{s} N={d} dot={d}\n", .{ @typeName(T), N, a.dot(b) });
        }
    }
}
