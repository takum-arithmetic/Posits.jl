# Posits.jl

[![Build Status](https://github.com/takum-arithmetic/Posits.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/takum-arithmetic/Posits.jl/actions/workflows/CI.yml?query=branch%3Amaster)

This package implements [posit arithmetic](https://posithub.org/docs/posit_standard-2.pdf), a tapered
precision machine number format. Eight new data types are defined, namely
`Posit8`, `Posit16`, `Posit32` and `Posit64` for the (linear) posits,
a floating-point format, and `LogPosit8`, `LogPosit16`,
`LogPosit32` and `LogPosit64` for the logarithmic posits, a logarithmic
number system. Internally the C99 library
[libposit](https://github.com/takum-arithmetic/libposit) is used for the
actual computations for the most part.

Using this package one is able to evaluate posits for real-world applications
in terms of precision. Given it is a software implementation the performance
is overall worse than the respective usual IEEE 754 floating-point hardware
implementations, but it is sufficient to evaluate the formats for reasonably
sized numerical benchmarks.

## Usage

The eight posit number types `Posit8`, `Posit16`, `Posit32`, `Posit64`
`LogPosit8`, `LogPosit16`, `LogPosit32`, `LogPosit64`
have been implemented to behave as much as any built-in floating-point
type. They are subtypes of `AbstractFloat` even though logarithmic posits itself
are strictly speaking not a floating-point number format, but a
logarithmic number system. However, as the majority of numerical code is
written to accept `AbstractFloat`s rather than `Real`s and logarithmic
posits share many properties of a typical floating-point number system,
this decision was made deliberately and for good reasons.

```julia
julia> using Posits

julia> x = Posit8(4.0)
Posit8(4.0)

julia> sqrt(x)
Posit8(2.117)
```

A wide range of functions are implemented for the number types. If you find
a floating-point function you need that is not yet implemented, please raise
an issue.

## Author and License

Posits.jl is developed by Laslo Hunhold and licensed under the ISC
license. See LICENSE for copyright and license details.
