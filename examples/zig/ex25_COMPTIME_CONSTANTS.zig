//! Same polynomial as Mojo polynomial[degree,True]; x is a runtime argument.
const std = @import("std");
fn polynomial(comptime degree: usize, x: f64) f64 {
    const coefficients = [_]f64{ 1, 2, 3, 4 };
    var value: f64 = 0;
    if (comptime degree == 2 or degree == 3) {
        inline for (0..degree + 1) |k| {
            value = value * x + coefficients[degree - k];
        }
    } else @compileError("only quadratic/cubic are demonstrated");
    return value;
}
test "runtime polynomial, compile-time degree and coefficients" {
    const x: f64 = 2;
    try std.testing.expectApproxEqAbs(@as(f64, 17), polynomial(2, x), 1e-13);
    try std.testing.expectApproxEqAbs(@as(f64, 49), polynomial(3, x), 1e-13);
    std.debug.print("quadratic=17 cubic=49\n", .{});
}
