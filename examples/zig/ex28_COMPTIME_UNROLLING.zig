//! Cubic Horner evaluation with compile-time versus runtime loop extent.
const std = @import("std");
fn polynomial(comptime degree: usize, comptime unroll: bool, x: f64) f64 {
    const coefficients = [_]f64{ 1, 2, 3, 4 };
    var value: f64 = 0;
    if (unroll) {
        inline for (0..degree + 1) |k| value = value * x + coefficients[degree - k];
    } else {
        for (0..degree + 1) |k| value = value * x + coefficients[degree - k];
    }
    return value;
}
test "unrolled and runtime-loop cubic" {
    const xs = [_]f64{ -1, 0.5, 2 };
    const expected = [_]f64{ -2, 3.25, 49 };
    for (xs, expected) |x, want| {
        const result = polynomial(3, true, x);
        try std.testing.expectApproxEqAbs(want, result, 1e-13);
        try std.testing.expectApproxEqAbs(result, polynomial(3, false, x), 1e-13);
        std.debug.print("x={d} unrolled={d} runtime_loop={d}\n", .{ x, result, polynomial(3, false, x) });
    }
}
