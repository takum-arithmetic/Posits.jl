module Posits

import Base: AbstractFloat, Int, Int8, Int16, Int32, Int64, Integer, MPFR,
	Signed, Unsigned, reinterpret
import Printf
import Random: rand, randexp, AbstractRNG, Sampler

using libposit_jll

export Posit8, Posit16, Posit32, Posit64, LogPosit8, LogPosit16,
       LogPosit32, LogPosit64, AnyPosit,
       NaRPosit8, NaRPosit16, NaRPosit32, NaRPosit64,
       NaRLogPosit8, NaRLogPosit16, NaRLogPosit32, NaRLogPosit64,
       posit, isnar

# type definitions
primitive type Posit8  <: AbstractFloat 8 end
primitive type Posit16 <: AbstractFloat 16 end
primitive type Posit32 <: AbstractFloat 32 end
primitive type Posit64 <: AbstractFloat 64 end

primitive type LogPosit8  <: AbstractFloat 8 end
primitive type LogPosit16 <: AbstractFloat 16 end
primitive type LogPosit32 <: AbstractFloat 32 end
primitive type LogPosit64 <: AbstractFloat 64 end

AnyPosit = Union{Posit8, Posit16, Posit32, Posit64, LogPosit8,
                 LogPosit16, LogPosit32, LogPosit64}
AnyPosit8 = Union{Posit8, LogPosit8}
AnyPosit16 = Union{Posit16, LogPosit16}
AnyPosit32 = Union{Posit32, LogPosit32}
AnyPosit64 = Union{Posit64, LogPosit64}

# NaR representations
const NaRPosit8  = Base.bitcast(Posit8,  0x80)
const NaRPosit16 = Base.bitcast(Posit16, 0x8000)
const NaRPosit32 = Base.bitcast(Posit32, 0x80000000)
const NaRPosit64 = Base.bitcast(Posit64, 0x8000000000000000)

const NaRLogPosit8  = Base.bitcast(LogPosit8,  0x80)
const NaRLogPosit16 = Base.bitcast(LogPosit16, 0x8000)
const NaRLogPosit32 = Base.bitcast(LogPosit32, 0x80000000)
const NaRLogPosit64 = Base.bitcast(LogPosit64, 0x8000000000000000)

# array of posit types with their corresponding integer types for
# metaprogramming within the module
posit_types = [
	(:Posit8,     "posit8",      :Int8),
	(:Posit16,    "posit16",     :Int16),
	(:Posit32,    "posit32",     :Int32),
	(:Posit64,    "posit64",     :Int64),
	(:LogPosit8,  "posit_log8",  :Int8),
	(:LogPosit16, "posit_log16", :Int16),
	(:LogPosit32, "posit_log32", :Int32),
	(:LogPosit64, "posit_log64", :Int64),
]

# integer reinterpret casts
Base.reinterpret(::Type{Unsigned}, x::AnyPosit8)  = reinterpret(UInt8, x)
Base.reinterpret(::Type{Unsigned}, x::AnyPosit16) = reinterpret(UInt16, x)
Base.reinterpret(::Type{Unsigned}, x::AnyPosit32) = reinterpret(UInt32, x)
Base.reinterpret(::Type{Unsigned}, x::AnyPosit64) = reinterpret(UInt64, x)

Base.reinterpret(::Type{Signed}, x::AnyPosit8)  = reinterpret(Int8, x)
Base.reinterpret(::Type{Signed}, x::AnyPosit16) = reinterpret(Int16, x)
Base.reinterpret(::Type{Signed}, x::AnyPosit32) = reinterpret(Int32, x)
Base.reinterpret(::Type{Signed}, x::AnyPosit64) = reinterpret(Int64, x)

Base.uinttype(::Type{<:AnyPosit8})  = UInt8
Base.uinttype(::Type{<:AnyPosit16}) = UInt16
Base.uinttype(::Type{<:AnyPosit32}) = UInt32
Base.uinttype(::Type{<:AnyPosit64}) = UInt64

@static if VERSION ≥ v"1.10"
	Base.inttype(::Type{<:AnyPosit8})  = Int8
	Base.inttype(::Type{<:AnyPosit16}) = Int16
	Base.inttype(::Type{<:AnyPosit32}) = Int32
	Base.inttype(::Type{<:AnyPosit64}) = Int64
end

# the only floating-point property that makes sense to implement for logarithmic posits is signbit()
Base.signbit(t::AnyPosit8)  = (reinterpret(Unsigned, t) & 0x80) !== 0x00
Base.signbit(t::AnyPosit16) = (reinterpret(Unsigned, t) & 0x8000) !== 0x0000
Base.signbit(t::AnyPosit32) = (reinterpret(Unsigned, t) & 0x80000000) !== 0x00000000
Base.signbit(t::AnyPosit64) = (reinterpret(Unsigned, t) & 0x8000000000000000) !== 0x0000000000000000

# left undefined are sign_mask, exponent_mask, exponent_one, exponent_half,
# significand_mask, exponent_bias, exponent_bits, significand_bits,
# decompose, for now also for posits
#
# TODO cleaner
Base.exponent(t::AnyPosit) = exponent(Float64(t))
Base.significand(t::AnyPosit) = ldexp(t, -Base.exponent(t))
Base.ldexp(t::AnyPosit, e::Integer) = t * ((typeof(t))(2) .^ e)
function Base.frexp(t::AnyPosit)
	exp = isnan(t) ? 0 : (Base.exponent(t) + 1)
	x = ldexp(t, -exp)
	return (x, exp)
end

Base.iszero(t::AnyPosit8)  = reinterpret(Unsigned, t) === 0x00
Base.iszero(t::AnyPosit16) = reinterpret(Unsigned, t) === 0x0000
Base.iszero(t::AnyPosit32) = reinterpret(Unsigned, t) === 0x00000000
Base.iszero(t::AnyPosit64) = reinterpret(Unsigned, t) === 0x0000000000000000

Base.isone(t::AnyPosit8)  = reinterpret(Unsigned, t) === 0x40
Base.isone(t::AnyPosit16) = reinterpret(Unsigned, t) === 0x4000
Base.isone(t::AnyPosit32) = reinterpret(Unsigned, t) === 0x40000000
Base.isone(t::AnyPosit64) = reinterpret(Unsigned, t) === 0x4000000000000000

Base.isfinite(t::AnyPosit8)  = reinterpret(Unsigned, t) !== 0x80
Base.isfinite(t::AnyPosit16) = reinterpret(Unsigned, t) !== 0x8000
Base.isfinite(t::AnyPosit32) = reinterpret(Unsigned, t) !== 0x80000000
Base.isfinite(t::AnyPosit64) = reinterpret(Unsigned, t) !== 0x8000000000000000

Base.isnan(t::AnyPosit8)  = reinterpret(Unsigned, t) === 0x80
Base.isnan(t::AnyPosit16) = reinterpret(Unsigned, t) === 0x8000
Base.isnan(t::AnyPosit32) = reinterpret(Unsigned, t) === 0x80000000
Base.isnan(t::AnyPosit64) = reinterpret(Unsigned, t) === 0x8000000000000000

isnar(t::AnyPosit8)  = Base.isnan(t)
isnar(t::AnyPosit16) = Base.isnan(t)
isnar(t::AnyPosit32) = Base.isnan(t)
isnar(t::AnyPosit64) = Base.isnan(t)

Base.issubnormal(t::AnyPosit) = false
Base.ispow2(t::Union{Posit8, Posit16, Posit32, Posit64}) = Base.isone(t)
Base.ispow2(t::Union{LogPosit8, LogPosit16, LogPosit32, LogPosit64}) = Base.ispow2(Float64(t))
Base.iseven(t::Union{Posit8, Posit16, Posit32, Posit64}) = Base.iszero(t)
Base.iseven(t::Union{LogPosit8, LogPosit16, LogPosit32, LogPosit64}) = Base.iseven(Float64(t))
Base.isodd(t::Union{Posit8, Posit16, Posit32, Posit64}) = Base.isone(t) || Base.isone(-t)
Base.isodd(t::Union{LogPosit8, LogPosit16, LogPosit32, LogPosit64}) = Base.isodd(Float64(t))

# precision
_mantissa_bit_count(t::Posit8)  = @ccall libposit.posit8_precision(reinterpret(Signed, t)::Int8)::UInt8
_mantissa_bit_count(t::Posit16) = @ccall libposit.posit16_precision(reinterpret(Signed, t)::Int16)::UInt8
_mantissa_bit_count(t::Posit32) = @ccall libposit.posit32_precision(reinterpret(Signed, t)::Int32)::UInt8
_mantissa_bit_count(t::Posit64) = @ccall libposit.posit64_precision(reinterpret(Signed, t)::Int64)::UInt8
_mantissa_bit_count(t::LogPosit8)  = @ccall libposit.posit_log8_precision(reinterpret(Signed, t)::Int8)::UInt8
_mantissa_bit_count(t::LogPosit16) = @ccall libposit.posit_log16_precision(reinterpret(Signed, t)::Int16)::UInt8
_mantissa_bit_count(t::LogPosit32) = @ccall libposit.posit_log32_precision(reinterpret(Signed, t)::Int32)::UInt8
_mantissa_bit_count(t::LogPosit64) = @ccall libposit.posit_log64_precision(reinterpret(Signed, t)::Int64)::UInt8

function Base.precision(t::AnyPosit; base::Integer = 2)
	base > 1 || throw(DomainError(base, "`base` cannot be less than 2."))
	m = 1 + _mantissa_bit_count(t)
	return base == 2 ? Int(m) : floor(Int, m / log2(base))
end

# For the types we determine the precision of the zero of said type, as this returns
# the worst-case precision, consistent with what you obtain with the respective
# IEEE 754 precision functions
Base.precision(T::Type{<:AnyPosit}; base::Integer = 2) = precision(zero(T); base)

# eps (follow definition; for the types it is simply eps(Posit_(1.0))
Base.eps(t::AnyPosit)     = max(t - prevfloat(t), nextfloat(t) - t)
Base.eps(::Type{Posit8})  = Base.bitcast(Posit8,  0x28)
Base.eps(::Type{Posit16}) = Base.bitcast(Posit16, 0x0a00)
Base.eps(::Type{Posit32}) = Base.bitcast(Posit32, 0x00a00000)
Base.eps(::Type{Posit64}) = Base.bitcast(Posit64, 0x0000a00000000000)
Base.eps(::Type{LogPosit8})  = Base.bitcast(LogPosit8,  0x24)
Base.eps(::Type{LogPosit16}) = Base.bitcast(LogPosit16, 0x08f1)
Base.eps(::Type{LogPosit32}) = Base.bitcast(LogPosit32, 0x008f1459)
Base.eps(::Type{LogPosit64}) = Base.bitcast(LogPosit64, 0x00008eb3a9f01975)

# rounding
Base.round(t::AnyPosit) = typeof(t)(Base.round(Float64(t)))

Base.round(t::AnyPosit, r::RoundingMode{:ToZero})  = typeof(t)(Base.trunc(Float64(t)))
Base.round(t::AnyPosit, r::RoundingMode{:Down})    = typeof(t)(Base.floor(Float64(t)))
Base.round(t::AnyPosit, r::RoundingMode{:Up})      = typeof(t)(Base.ceil(Float64(t)))
Base.round(t::AnyPosit, r::RoundingMode{:Nearest}) = typeof(t)(Base.round(Float64(t)))

Base.trunc(t::AnyPosit) = Base.signbit(t) ? Base.ceil(t) : Base.floor(t)
Base.trunc(::Type{T}, t::AnyPosit) where {T <: Integer} = Base.trunc(T, Float64(t))

# type limits
Base.typemin(::Type{Posit8})  = NaRPosit8
Base.typemin(::Type{Posit16}) = NaRPosit16
Base.typemin(::Type{Posit32}) = NaRPosit32
Base.typemin(::Type{Posit64}) = NaRPosit64
Base.typemin(::Type{LogPosit8})  = NaRLogPosit8
Base.typemin(::Type{LogPosit16}) = NaRLogPosit16
Base.typemin(::Type{LogPosit32}) = NaRLogPosit32
Base.typemin(::Type{LogPosit64}) = NaRLogPosit64

Base.typemax(T::Type{<:AnyPosit8})  = Base.bitcast(T, 0x7f)
Base.typemax(T::Type{<:AnyPosit16}) = Base.bitcast(T, 0x7fff)
Base.typemax(T::Type{<:AnyPosit32}) = Base.bitcast(T, 0x7fffffff)
Base.typemax(T::Type{<:AnyPosit64}) = Base.bitcast(T, 0x7fffffffffffffff)

Base.floatmin(T::Type{<:AnyPosit8})  = Base.bitcast(T, 0x01)
Base.floatmin(T::Type{<:AnyPosit16}) = Base.bitcast(T, 0x0001)
Base.floatmin(T::Type{<:AnyPosit32}) = Base.bitcast(T, 0x00000001)
Base.floatmin(T::Type{<:AnyPosit64}) = Base.bitcast(T, 0x0000000000000001)

Base.floatmax(T::Type{<:AnyPosit8})  = Base.bitcast(T, 0x7f)
Base.floatmax(T::Type{<:AnyPosit16}) = Base.bitcast(T, 0x7fff)
Base.floatmax(T::Type{<:AnyPosit32}) = Base.bitcast(T, 0x7fffffff)
Base.floatmax(T::Type{<:AnyPosit64}) = Base.bitcast(T, 0x7fffffffffffffff)

Base.maxintfloat(::Type{T}) where {T<:Union{Posit8, Posit16, Posit32, Posit64}} = T(1.0)
Base.maxintfloat(::Type{LogPosit8}) = LogPosit8(2.0 ^ 3)
Base.maxintfloat(::Type{LogPosit16}) = LogPosit16(2.0 ^ 9)
Base.maxintfloat(::Type{LogPosit32}) = LogPosit32(2.0 ^ 24)
Base.maxintfloat(::Type{LogPosit64}) = LogPosit64(2.0 ^ 55)

# conversions from floating-point
Posit8(f::Float16) = Base.bitcast(Posit8, @ccall libposit.posit8_from_float32(Float32(f)::Float32)::Int8)
Posit8(f::Float32) = Base.bitcast(Posit8, @ccall libposit.posit8_from_float32(f::Float32)::Int8)
Posit8(f::Float64) = Base.bitcast(Posit8, @ccall libposit.posit8_from_float64(f::Float64)::Int8)
LogPosit8(f::Float16) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_float32(Float32(f)::Float32)::Int8)
LogPosit8(f::Float32) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_float32(f::Float32)::Int8)
LogPosit8(f::Float64) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_float64(f::Float64)::Int8)

Posit16(f::Float16) = Base.bitcast(Posit16, @ccall libposit.posit16_from_float32(Float32(f)::Float32)::Int16)
Posit16(f::Float32) = Base.bitcast(Posit16, @ccall libposit.posit16_from_float32(f::Float32)::Int16)
Posit16(f::Float64) = Base.bitcast(Posit16, @ccall libposit.posit16_from_float64(f::Float64)::Int16)
LogPosit16(f::Float16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_float32(Float32(f)::Float32)::Int16)
LogPosit16(f::Float32) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_float32(f::Float32)::Int16)
LogPosit16(f::Float64) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_float64(f::Float64)::Int16)

Posit32(f::Float16) = Base.bitcast(Posit32, @ccall libposit.posit32_from_float32(Float32(f)::Float32)::Int32)
Posit32(f::Float32) = Base.bitcast(Posit32, @ccall libposit.posit32_from_float32(f::Float32)::Int32)
Posit32(f::Float64) = Base.bitcast(Posit32, @ccall libposit.posit32_from_float64(f::Float64)::Int32)
LogPosit32(f::Float16) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_float32(Float32(f)::Float32)::Int32)
LogPosit32(f::Float32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_float32(f::Float32)::Int32)
LogPosit32(f::Float64) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_float64(f::Float64)::Int32)

Posit64(f::Float16) = Base.bitcast(Posit64, @ccall libposit.posit64_from_float32(Float32(f)::Float32)::Int64)
Posit64(f::Float32) = Base.bitcast(Posit64, @ccall libposit.posit64_from_float32(f::Float32)::Int64)
Posit64(f::Float64) = Base.bitcast(Posit64, @ccall libposit.posit64_from_float64(f::Float64)::Int64)
LogPosit64(f::Float16) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_float32(Float32(f)::Float32)::Int64)
LogPosit64(f::Float32) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_float32(f::Float32)::Int64)
LogPosit64(f::Float64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_float64(f::Float64)::Int64)

# conversion from integers with promote rules
Posit8(i::Integer)  = Base.convert(Posit8,  Base.convert(Float64, i))
Posit16(i::Integer) = Base.convert(Posit16, Base.convert(Float64, i))
Posit32(i::Integer) = Base.convert(Posit32, Base.convert(Float64, i))
Posit64(i::Integer) = Base.convert(Posit64, Base.convert(Float64, i))
LogPosit8(i::Integer)  = Base.convert(LogPosit8,  Base.convert(Float64, i))
LogPosit16(i::Integer) = Base.convert(LogPosit16, Base.convert(Float64, i))
LogPosit32(i::Integer) = Base.convert(LogPosit32, Base.convert(Float64, i))
LogPosit64(i::Integer) = Base.convert(LogPosit64, Base.convert(Float64, i))
Base.promote_rule(T::Type{<:AnyPosit}, ::Type{<:Integer}) = T

# conversion from BigFloat
Posit8(f::BigFloat) = Posit8(Float64(f))
Posit16(f::BigFloat) = Posit16(Float64(f))
Posit32(f::BigFloat) = Posit32(Float64(f))
LogPosit8(f::BigFloat) = LogPosit8(Float64(f))
LogPosit16(f::BigFloat) = LogPosit16(Float64(f))
LogPosit32(f::BigFloat) = LogPosit32(Float64(f))

function Posit64(f::BigFloat)
	if iszero(f)
		# truncated() is buggy when the input is zero
		return zero(Posit64)
	else
		# take the absolute value so we are only working with
		# positive values, as only then can we directly transplant
		# the fraction bits
		fabs = abs(f)

		# first round the BigFloat to the maximum possible number
		# of posit fraction bits, namely 1 + (64 - 5) = 60, as
		# this possibly rounds the value and lets us obtain the
		# exponent for further processing
		f60 = BigFloat(fabs; precision = 60)

		# Build a raw posit with f60's exponent
		rawp = ldexp(one(Posit64), exponent(f60))

		# Obtain the number of fraction bits the raw posit can
		# represent and round f60 down to this number of bits
		fprec = BigFloat(f; precision = precision(rawp))

		# During this process, fprec might have been rounded
		# insofar that the exponent changed, which would require
		# us to redo the whole process. We do it unconditionally,
		# so we take the exponent of fprec, update the raw
		# posit and round fprec accordingly to its capacity
		rawp = ldexp(one(Posit64), exponent(fprec))
		fprec = BigFloat(fprec; precision = precision(rawp))

		# Now we have the raw posit, which we just need to fill
		# with fprec's fraction bits. For this we use an obscure
		# internal MPFR function and also unset the highest
		# fraction bit, as it is stored implicitly in posits
		fraction = Base.MPFR.truncated(UInt64, fprec.d::MPFR.BigFloatData, precision(rawp))
		fraction &= ~(UInt64(2) ^ (precision(rawp) - 1))

		# Combine raw posit and fraction
		result = Base.bitcast(Posit64, Base.bitcast(UInt64, rawp) | fraction)

		# Return the result, negating it if f was negative
		return (f < 0) ? -result : result
	end
end

# TODO Conversion from BigFloat to LogPosit64

# conversion from irrational numbers
Posit8(i::AbstractIrrational) = Posit8(Float64(i))
Posit16(i::AbstractIrrational) = Posit16(Float64(i))
Posit32(i::AbstractIrrational) = Posit32(Float64(i))
Posit64(i::AbstractIrrational) = Posit64(BigFloat(i; precision = 60))
LogPosit8(i::AbstractIrrational) = LogPosit8(Float64(i))
LogPosit16(i::AbstractIrrational) = LogPosit16(Float64(i))
LogPosit32(i::AbstractIrrational) = LogPosit32(Float64(i))
LogPosit64(i::AbstractIrrational) = LogPosit64(Float64(i)) # TODO

# conversions to floating-point
Base.Float16(t::Posit8)  = Float16(@ccall libposit.posit8_to_float32(reinterpret(Signed, t)::Int8)::Float32)
Base.Float16(t::Posit16) = Float16(@ccall libposit.posit16_to_float32(reinterpret(Signed, t)::Int16)::Float32)
Base.Float16(t::Posit32) = Float16(@ccall libposit.posit32_to_float32(reinterpret(Signed, t)::Int32)::Float32)
Base.Float16(t::Posit64) = Float16(@ccall libposit.posit64_to_float32(reinterpret(Signed, t)::Int64)::Float32)
Base.Float16(t::LogPosit8)  = Float16(@ccall libposit.posit_log8_to_float32(reinterpret(Signed, t)::Int8)::Float32)
Base.Float16(t::LogPosit16) = Float16(@ccall libposit.posit_log16_to_float32(reinterpret(Signed, t)::Int16)::Float32)
Base.Float16(t::LogPosit32) = Float16(@ccall libposit.posit_log32_to_float32(reinterpret(Signed, t)::Int32)::Float32)
Base.Float16(t::LogPosit64) = Float16(@ccall libposit.posit_log64_to_float32(reinterpret(Signed, t)::Int64)::Float32)
Base.Float32(t::Posit8)  = @ccall libposit.posit8_to_float32(reinterpret(Signed, t)::Int8)::Float32
Base.Float32(t::Posit16) = @ccall libposit.posit16_to_float32(reinterpret(Signed, t)::Int16)::Float32
Base.Float32(t::Posit32) = @ccall libposit.posit32_to_float32(reinterpret(Signed, t)::Int32)::Float32
Base.Float32(t::Posit64) = @ccall libposit.posit64_to_float32(reinterpret(Signed, t)::Int64)::Float32
Base.Float32(t::LogPosit8)  = @ccall libposit.posit_log8_to_float32(reinterpret(Signed, t)::Int8)::Float32
Base.Float32(t::LogPosit16) = @ccall libposit.posit_log16_to_float32(reinterpret(Signed, t)::Int16)::Float32
Base.Float32(t::LogPosit32) = @ccall libposit.posit_log32_to_float32(reinterpret(Signed, t)::Int32)::Float32
Base.Float32(t::LogPosit64) = @ccall libposit.posit_log64_to_float32(reinterpret(Signed, t)::Int64)::Float32
Base.Float64(t::Posit8)  = @ccall libposit.posit8_to_float64(reinterpret(Signed, t)::Int8)::Float64
Base.Float64(t::Posit16) = @ccall libposit.posit16_to_float64(reinterpret(Signed, t)::Int16)::Float64
Base.Float64(t::Posit32) = @ccall libposit.posit32_to_float64(reinterpret(Signed, t)::Int32)::Float64
Base.Float64(t::Posit64) = @ccall libposit.posit64_to_float64(reinterpret(Signed, t)::Int64)::Float64
Base.Float64(t::LogPosit8)  = @ccall libposit.posit_log8_to_float64(reinterpret(Signed, t)::Int8)::Float64
Base.Float64(t::LogPosit16) = @ccall libposit.posit_log16_to_float64(reinterpret(Signed, t)::Int16)::Float64
Base.Float64(t::LogPosit32) = @ccall libposit.posit_log32_to_float64(reinterpret(Signed, t)::Int32)::Float64
Base.Float64(t::LogPosit64) = @ccall libposit.posit_log64_to_float64(reinterpret(Signed, t)::Int64)::Float64

# conversion to integer
Base.unsafe_trunc(T::Type{<:Integer}, t::AnyPosit)  = Base.unsafe_trunc(T, Float64(t))
Base.round(I::Type{<:Integer}, t::AnyPosit) = Base.round(I, Float64(t))

function (::Type{I})(t::Union{Posit8, Posit16, Posit32, Posit64}) where I <: Integer
	if t == -1
		return I(-1)
	elseif t == 0
		return I(0)
	elseif t == 1
		return I(1)
	else
		throw(InexactError(:round, I, t))
	end
end

function (::Type{I})(t::Union{LogPosit8, LogPosit16, LogPosit32, LogPosit64}) where I <: Integer
	# TODO test if integer is representable and return an InexactError if not
	return I(Float64(t))
end

# inter-posit conversions
Posit8(t::Posit16) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit16(reinterpret(Signed, t)::Int16)::Int8)
Posit8(t::Posit32) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit32(reinterpret(Signed, t)::Int32)::Int8)
Posit8(t::Posit64) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit64(reinterpret(Signed, t)::Int64)::Int8)
Posit8(t::LogPosit8) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit_log8(reinterpret(Signed, t)::Int8)::Int8)
Posit8(t::LogPosit16) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit_log16(reinterpret(Signed, t)::Int16)::Int8)
Posit8(t::LogPosit32) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit_log32(reinterpret(Signed, t)::Int32)::Int8)
Posit8(t::LogPosit64) = Base.bitcast(Posit8, @ccall libposit.posit8_from_posit_log64(reinterpret(Signed, t)::Int64)::Int8)

Posit16(t::Posit8)  = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit8(reinterpret(Signed, t)::Int16)::Int16)
Posit16(t::Posit32) = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit32(reinterpret(Signed, t)::Int32)::Int16)
Posit16(t::Posit64) = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit64(reinterpret(Signed, t)::Int64)::Int16)
Posit16(t::LogPosit8) = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit_log8(reinterpret(Signed, t)::Int8)::Int16)
Posit16(t::LogPosit16) = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit_log16(reinterpret(Signed, t)::Int16)::Int16)
Posit16(t::LogPosit32) = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit_log32(reinterpret(Signed, t)::Int32)::Int16)
Posit16(t::LogPosit64) = Base.bitcast(Posit16, @ccall libposit.posit16_from_posit_log64(reinterpret(Signed, t)::Int64)::Int16)

Posit32(t::Posit8)  = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit8(reinterpret(Signed, t)::Int8)::Int32)
Posit32(t::Posit16) = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit16(reinterpret(Signed, t)::Int16)::Int32)
Posit32(t::Posit64) = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit64(reinterpret(Signed, t)::Int64)::Int32)
Posit32(t::LogPosit8) = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit_log8(reinterpret(Signed, t)::Int8)::Int32)
Posit32(t::LogPosit16) = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit_log16(reinterpret(Signed, t)::Int16)::Int32)
Posit32(t::LogPosit32) = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit_log32(reinterpret(Signed, t)::Int32)::Int32)
Posit32(t::LogPosit64) = Base.bitcast(Posit32, @ccall libposit.posit32_from_posit_log64(reinterpret(Signed, t)::Int64)::Int32)

Posit64(t::Posit8)  = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit8(reinterpret(Signed, t)::Int8)::Int64)
Posit64(t::Posit16) = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit16(reinterpret(Signed, t)::Int16)::Int64)
Posit64(t::Posit32) = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit32(reinterpret(Signed, t)::Int32)::Int64)
Posit64(t::LogPosit8) = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit_log8(reinterpret(Signed, t)::Int8)::Int64)
Posit64(t::LogPosit16) = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit_log16(reinterpret(Signed, t)::Int16)::Int64)
Posit64(t::LogPosit32) = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit_log32(reinterpret(Signed, t)::Int32)::Int64)
Posit64(t::LogPosit64) = Base.bitcast(Posit64, @ccall libposit.posit64_from_posit_log64(reinterpret(Signed, t)::Int64)::Int64)

LogPosit8(t::LogPosit16) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit_log16(reinterpret(Signed, t)::Int16)::Int8)
LogPosit8(t::LogPosit32) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit_log32(reinterpret(Signed, t)::Int32)::Int8)
LogPosit8(t::LogPosit64) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit_log64(reinterpret(Signed, t)::Int64)::Int8)
LogPosit8(t::Posit8) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit8(reinterpret(Signed, t)::Int8)::Int8)
LogPosit8(t::Posit16) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit16(reinterpret(Signed, t)::Int16)::Int8)
LogPosit8(t::Posit32) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit32(reinterpret(Signed, t)::Int32)::Int8)
LogPosit8(t::Posit64) = Base.bitcast(LogPosit8, @ccall libposit.posit_log8_from_posit64(reinterpret(Signed, t)::Int64)::Int8)

LogPosit16(t::LogPosit8)  = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit_log8(reinterpret(Signed, t)::Int8)::Int16)
LogPosit16(t::LogPosit32) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit_log32(reinterpret(Signed, t)::Int32)::Int16)
LogPosit16(t::LogPosit64) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit_log64(reinterpret(Signed, t)::Int64)::Int16)
LogPosit16(t::Posit8) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit8(reinterpret(Signed, t)::Int8)::Int16)
LogPosit16(t::Posit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit16(reinterpret(Signed, t)::Int16)::Int16)
LogPosit16(t::Posit32) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit32(reinterpret(Signed, t)::Int32)::Int16)
LogPosit16(t::Posit64) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_from_posit64(reinterpret(Signed, t)::Int64)::Int16)

LogPosit32(t::LogPosit8)  = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit_log8(reinterpret(Signed, t)::Int8)::Int32)
LogPosit32(t::LogPosit16) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit_log16(reinterpret(Signed, t)::Int16)::Int32)
LogPosit32(t::LogPosit64) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit_log64(reinterpret(Signed, t)::Int64)::Int32)
LogPosit32(t::Posit8) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit8(reinterpret(Signed, t)::Int8)::Int32)
LogPosit32(t::Posit16) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit16(reinterpret(Signed, t)::Int16)::Int32)
LogPosit32(t::Posit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit32(reinterpret(Signed, t)::Int32)::Int32)
LogPosit32(t::Posit64) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_from_posit64(reinterpret(Signed, t)::Int64)::Int32)

LogPosit64(t::LogPosit8)  = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit_log8(reinterpret(Signed, t)::Int8)::Int64)
LogPosit64(t::LogPosit16) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit_log16(reinterpret(Signed, t)::Int16)::Int64)
LogPosit64(t::LogPosit32) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit_log32(reinterpret(Signed, t)::Int32)::Int64)
LogPosit64(t::Posit8)  = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit8(reinterpret(Signed, t)::Int8)::Int64)
LogPosit64(t::Posit16) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit16(reinterpret(Signed, t)::Int16)::Int64)
LogPosit64(t::Posit32) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit32(reinterpret(Signed, t)::Int32)::Int64)
LogPosit64(t::Posit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_from_posit64(reinterpret(Signed, t)::Int64)::Int64)

# inter-posit promote rules
Base.promote_rule(::Type{Posit16}, ::Type{Posit8}) = Posit16
Base.promote_rule(::Type{LogPosit16}, ::Type{LogPosit8}) = LogPosit16

Base.promote_rule(::Type{Posit32}, ::Type{Posit8}) = Posit32
Base.promote_rule(::Type{Posit32}, ::Type{Posit16}) = Posit32
Base.promote_rule(::Type{LogPosit32}, ::Type{LogPosit8}) = LogPosit32
Base.promote_rule(::Type{LogPosit32}, ::Type{LogPosit16}) = LogPosit32

Base.promote_rule(::Type{Posit64}, ::Type{Posit8}) = Posit64
Base.promote_rule(::Type{Posit64}, ::Type{Posit16}) = Posit64
Base.promote_rule(::Type{Posit64}, ::Type{Posit32}) = Posit64
Base.promote_rule(::Type{LogPosit64}, ::Type{LogPosit8}) = LogPosit64
Base.promote_rule(::Type{LogPosit64}, ::Type{LogPosit16}) = LogPosit64
Base.promote_rule(::Type{LogPosit64}, ::Type{LogPosit32}) = LogPosit64

# IEEE 754 floating point promote rules, where we expand all to Float64
# given we would otherwise constrain the dynamic range
Base.promote_rule(::Type{Float16}, ::Type{<:AnyPosit}) = Float64
Base.promote_rule(::Type{Float32}, ::Type{<:AnyPosit}) = Float64
Base.promote_rule(::Type{Float64}, ::Type{<:AnyPosit}) = Float64

# arithmetic
Base.:(+)(x::Posit8,  y::Posit8)  = Base.bitcast(Posit8,  @ccall libposit.posit8_addition(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(+)(x::Posit16, y::Posit16) = Base.bitcast(Posit16, @ccall libposit.posit16_addition(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(+)(x::Posit32, y::Posit32) = Base.bitcast(Posit32, @ccall libposit.posit32_addition(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(+)(x::Posit64, y::Posit64) = Base.bitcast(Posit64, @ccall libposit.posit64_addition(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)
Base.:(+)(x::LogPosit8,  y::LogPosit8)  = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_addition(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(+)(x::LogPosit16, y::LogPosit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_addition(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(+)(x::LogPosit32, y::LogPosit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_addition(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(+)(x::LogPosit64, y::LogPosit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_addition(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)

Base.:(-)(x::Posit8,  y::Posit8)  = Base.bitcast(Posit8,  @ccall libposit.posit8_subtraction(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(-)(x::Posit16, y::Posit16) = Base.bitcast(Posit16, @ccall libposit.posit16_subtraction(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(-)(x::Posit32, y::Posit32) = Base.bitcast(Posit32, @ccall libposit.posit32_subtraction(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(-)(x::Posit64, y::Posit64) = Base.bitcast(Posit64, @ccall libposit.posit64_subtraction(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)
Base.:(-)(x::LogPosit8,  y::LogPosit8)  = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_subtraction(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(-)(x::LogPosit16, y::LogPosit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_subtraction(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(-)(x::LogPosit32, y::LogPosit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_subtraction(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(-)(x::LogPosit64, y::LogPosit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_subtraction(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)

Base.:(*)(x::Posit8,  y::Posit8)  = Base.bitcast(Posit8,  @ccall libposit.posit8_multiplication(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(*)(x::Posit16, y::Posit16) = Base.bitcast(Posit16, @ccall libposit.posit16_multiplication(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(*)(x::Posit32, y::Posit32) = Base.bitcast(Posit32, @ccall libposit.posit32_multiplication(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(*)(x::Posit64, y::Posit64) = Base.bitcast(Posit64, @ccall libposit.posit64_multiplication(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)
Base.:(*)(x::LogPosit8,  y::LogPosit8)  = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_multiplication(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(*)(x::LogPosit16, y::LogPosit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_multiplication(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(*)(x::LogPosit32, y::LogPosit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_multiplication(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(*)(x::LogPosit64, y::LogPosit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_multiplication(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)

Base.:(/)(x::Posit8,  y::Posit8)  = Base.bitcast(Posit8,  @ccall libposit.posit8_division(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(/)(x::Posit16, y::Posit16) = Base.bitcast(Posit16, @ccall libposit.posit16_division(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(/)(x::Posit32, y::Posit32) = Base.bitcast(Posit32, @ccall libposit.posit32_division(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(/)(x::Posit64, y::Posit64) = Base.bitcast(Posit64, @ccall libposit.posit64_division(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)
Base.:(/)(x::LogPosit8,  y::LogPosit8)  = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_division(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(/)(x::LogPosit16, y::LogPosit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_division(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(/)(x::LogPosit32, y::LogPosit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_division(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(/)(x::LogPosit64, y::LogPosit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_division(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)

Base.:(^)(x::Posit8,  y::Posit8)  = Base.bitcast(Posit8,  @ccall libposit.posit8_power(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(^)(x::Posit16, y::Posit16) = Base.bitcast(Posit16, @ccall libposit.posit16_power(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(^)(x::Posit32, y::Posit32) = Base.bitcast(Posit32, @ccall libposit.posit32_power(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(^)(x::Posit64, y::Posit64) = Base.bitcast(Posit64, @ccall libposit.posit64_power(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)
Base.:(^)(x::LogPosit8,  y::LogPosit8)  = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_power(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.:(^)(x::LogPosit16, y::LogPosit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_power(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.:(^)(x::LogPosit32, y::LogPosit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_power(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.:(^)(x::LogPosit64, y::LogPosit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_power(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)

Base.:(^)(x::Posit8,  n::Integer) = Base.bitcast(Posit8,  @ccall libposit.posit8_integer_power(reinterpret(Signed, x)::Int8, n::Int64)::Int8)
Base.:(^)(x::Posit16, n::Integer) = Base.bitcast(Posit16, @ccall libposit.posit16_integer_power(reinterpret(Signed, x)::Int16, n::Int64)::Int16)
Base.:(^)(x::Posit32, n::Integer) = Base.bitcast(Posit32, @ccall libposit.posit32_integer_power(reinterpret(Signed, x)::Int32, n::Int64)::Int32)
Base.:(^)(x::Posit64, n::Integer) = Base.bitcast(Posit64, @ccall libposit.posit64_integer_power(reinterpret(Signed, x)::Int64, n::Int64)::Int64)
Base.:(^)(x::LogPosit8,  n::Integer) = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_integer_power(reinterpret(Signed, x)::Int8, n::Int64)::Int8)
Base.:(^)(x::LogPosit16, n::Integer) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_integer_power(reinterpret(Signed, x)::Int16, n::Int64)::Int16)
Base.:(^)(x::LogPosit32, n::Integer) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_integer_power(reinterpret(Signed, x)::Int32, n::Int64)::Int32)
Base.:(^)(x::LogPosit64, n::Integer) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_integer_power(reinterpret(Signed, x)::Int64, n::Int64)::Int64)

Base.:(-)(t::AnyPosit) = Base.bitcast(typeof(t), -reinterpret(Signed, t))

Base.zero(T::Type{<:AnyPosit8})  = Base.bitcast(T, 0x00)
Base.zero(T::Type{<:AnyPosit16}) = Base.bitcast(T, 0x0000)
Base.zero(T::Type{<:AnyPosit32}) = Base.bitcast(T, 0x00000000)
Base.zero(T::Type{<:AnyPosit64}) = Base.bitcast(T, 0x0000000000000000)

Base.one(T::Type{<:AnyPosit8})  = Base.bitcast(T, 0x40)
Base.one(T::Type{<:AnyPosit16}) = Base.bitcast(T, 0x4000)
Base.one(T::Type{<:AnyPosit32}) = Base.bitcast(T, 0x40000000)
Base.one(T::Type{<:AnyPosit64}) = Base.bitcast(T, 0x4000000000000000)

Base.inv(t::Posit8) = Base.bitcast(Posit8,  @ccall libposit.posit8_inversion(reinterpret(Signed, t)::Int8)::Int8)
Base.inv(t::Posit16) = Base.bitcast(Posit16,  @ccall libposit.posit16_inversion(reinterpret(Signed, t)::Int16)::Int16)
Base.inv(t::Posit32) = Base.bitcast(Posit32,  @ccall libposit.posit32_inversion(reinterpret(Signed, t)::Int32)::Int32)
Base.inv(t::Posit64) = Base.bitcast(Posit64,  @ccall libposit.posit64_inversion(reinterpret(Signed, t)::Int64)::Int64)
Base.inv(t::LogPosit8) = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_inversion(reinterpret(Signed, t)::Int8)::Int8)
Base.inv(t::LogPosit16) = Base.bitcast(LogPosit16,  @ccall libposit.posit_log16_inversion(reinterpret(Signed, t)::Int16)::Int16)
Base.inv(t::LogPosit32) = Base.bitcast(LogPosit32,  @ccall libposit.posit_log32_inversion(reinterpret(Signed, t)::Int32)::Int32)
Base.inv(t::LogPosit64) = Base.bitcast(LogPosit64,  @ccall libposit.posit_log64_inversion(reinterpret(Signed, t)::Int64)::Int64)

# comparisons
Base.:(==)(x::T, y::T) where {T <: AnyPosit} = reinterpret(Signed, x) == reinterpret(Signed, y)
Base.:(!=)(x::T, y::T) where {T <: AnyPosit} = reinterpret(Signed, x) != reinterpret(Signed, y)
Base.:(<)(x::T, y::T) where {T <: AnyPosit} = reinterpret(Signed, x) < reinterpret(Signed, y)
Base.:(<=)(x::T, y::T) where {T <: AnyPosit} = reinterpret(Signed, x) <= reinterpret(Signed, y)

Base.isequal(x::T, y::T) where {T <: AnyPosit} = (x == y)

# TODO: widen for 64-bit to BigFloat
Base.widen(::Type{Posit8}) = Posit16
Base.widen(::Type{Posit16}) = Posit32
Base.widen(::Type{Posit32}) = Posit64
Base.widen(::Type{LogPosit8}) = LogPosit16
Base.widen(::Type{LogPosit16}) = LogPosit32
Base.widen(::Type{LogPosit32}) = LogPosit64

Base.widemul(x::Union{Posit8, Posit16, Posit32}, y::Union{Posit8, Posit16, Posit32}) = Basen.widen(x) * Base.widen(y)
Base.widemul(x::Union{LogPosit8, LogPosit16, LogPosit32}, y::Union{LogPosit8, LogPosit16, LogPosit32}) = Basen.widen(x) * Base.widen(y)

# output
function Base.show(io::IO, t::AnyPosit)
	has_type_info = typeof(t) === Base.get(io, :typeinfo, Any)
	if isnar(t)
		Base.print(io, "NaR", string(typeof(t)))
	else
		has_type_info || Base.print(io, string(typeof(t)) * "(")
		@static if VERSION ≥ v"1.7"
			@Printf.printf(IOContext(io, :typeinfo=>typeof(t)), "%.*g", max(4, 1 + Base.precision(t; base = 10)), Float64(t))
		else
			@Printf.printf(IOContext(io, :typeinfo=>typeof(t)), "%f", Float64(t))
		end
		has_type_info || Base.print(io, ")")
	end
end

Printf.tofloat(t::AnyPosit) = Float64(t)

# bitstring
Base.bitstring(t::AnyPosit) = Base.bitstring(reinterpret(Unsigned, t))

# next and previous number
function Base.nextfloat(t::AnyPosit, n::Integer)
	# preserve NaR
	if isnar(t)
		return t
	end

	# convert t to its integral representation
	t_integer = reinterpret(Signed, t)

	# set result to NaR and catch valid cases later
	result_integer = typemin(t_integer)

	# check if adding n would under- or overflow, return NaR otherwise
	if (n >= 0 && t_integer <= reinterpret(Signed, typemax(t)) - n) ||
	   (n < 0 && t_integer >= reinterpret(Signed, typemin(t)) + (-n))
		result_integer = typeof(t_integer)(t_integer + n)
	end

	return Base.bitcast(typeof(t), result_integer)
end
Base.prevfloat(t::AnyPosit, n::Integer) = Base.nextfloat(t, -n)

Base.nextfloat(t::AnyPosit) = Base.nextfloat(t, 1)
Base.prevfloat(t::AnyPosit) = Base.prevfloat(t, 1)

# math functions
Base.abs(t::AnyPosit) = (t < 0) ? -t : t
Base.abs2(t::AnyPosit) = t * t

# 2-argument arctangent
Base.atan(x::Posit8,  y::Posit8)  = Base.bitcast(Posit8,  @ccall libposit.posit8_arctan2(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.atan(x::Posit16, y::Posit16) = Base.bitcast(Posit16, @ccall libposit.posit16_arctan2(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.atan(x::Posit32, y::Posit32) = Base.bitcast(Posit32, @ccall libposit.posit32_arctan2(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.atan(x::Posit64, y::Posit64) = Base.bitcast(Posit64, @ccall libposit.posit64_arctan2(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)
Base.atan(x::LogPosit8,  y::LogPosit8)  = Base.bitcast(LogPosit8,  @ccall libposit.posit_log8_arctan2(reinterpret(Signed, x)::Int8, reinterpret(Signed, y)::Int8)::Int8)
Base.atan(x::LogPosit16, y::LogPosit16) = Base.bitcast(LogPosit16, @ccall libposit.posit_log16_arctan2(reinterpret(Signed, x)::Int16, reinterpret(Signed, y)::Int16)::Int16)
Base.atan(x::LogPosit32, y::LogPosit32) = Base.bitcast(LogPosit32, @ccall libposit.posit_log32_arctan2(reinterpret(Signed, x)::Int32, reinterpret(Signed, y)::Int32)::Int32)
Base.atan(x::LogPosit64, y::LogPosit64) = Base.bitcast(LogPosit64, @ccall libposit.posit_log64_arctan2(reinterpret(Signed, x)::Int64, reinterpret(Signed, y)::Int64)::Int64)

math_functions = [
	(:acos,  :arccos),
	(:acosh, :arcosh),
	(:acot,  :arccot),
	(:acoth, :arcoth),
	(:acsc,  :arccsc),
	(:acsch, :arcsch),
	(:asec,  :arcsec),
	(:asech, :arsech),
	(:asin,  :arcsin),
	(:asinh, :arsinh),
	(:atan,  :arctan),
	(:atanh, :artanh),
	(:cbrt,  :root, :(3::Int64)),
	(:cos,   :cos),
	(:cospi, :cos_pi_times),
	(:cosh,  :cosh),
	(:cot,   :cot),
	(:coth,  :coth),
	(:csc,   :csc),
	(:csch,  :csch),
	(:exp,   :exp),
	(:exp10, :(10_raised)),
	(:exp2,  :(2_raised)),
	(:expm1, :exp_minus_1),
	(:log,   :ln),
	(:log10, :lg),
	(:log1p, :ln_1_plus),
	(:log2,  :lb),
	(:sec,   :sec),
	(:sech,  :sech),
	(:sin,   :sin),
	(:sinpi, :sin_pi_times),
	(:sinh,  :sinh),
	(:sqrt,  :square_root),
	(:tan,   :tan),
	(:tanh,  :tanh),
]

@static if VERSION ≥ v"1.10"
	push!(math_functions, (:tanpi, :tan_pi_times))
end

for (posit_type, posit_type_cname, posit_integer_type) in posit_types
	for (math_function, library_math_function, arguments...) in math_functions
		@eval begin
			Base.$math_function(t::$posit_type) = Base.bitcast($posit_type,
				@ccall libposit.$(Symbol(posit_type_cname, "_", library_math_function))(
				reinterpret(Signed, t)::$posit_integer_type, $(arguments...))::$posit_integer_type)
		end
	end
end

# random
rand(rng::AbstractRNG, ::Sampler{T}) where {T <: AnyPosit8} = T(rand(rng, Float64))
rand(rng::AbstractRNG, ::Sampler{T}) where {T <: AnyPosit16} = T(rand(rng, Float64))
rand(rng::AbstractRNG, ::Sampler{T}) where {T <: AnyPosit32} = T(rand(rng, Float64))
rand(rng::AbstractRNG, ::Sampler{Posit64}) = setprecision(BigFloat, 60) do; Posit64(rand(rng, BigFloat)) end
rand(rng::AbstractRNG, ::Sampler{LogPosit64}) = LogPosit64(rand(rng, Float64)) # TODO

randexp(rng::AbstractRNG, ::Sampler{T}) where {T <: AnyPosit8} = T(randexp(rng, Float64))
randexp(rng::AbstractRNG, ::Sampler{T}) where {T <: AnyPosit16} = T(randexp(rng, Float64))
randexp(rng::AbstractRNG, ::Sampler{T}) where {T <: AnyPosit32} = T(randexp(rng, Float64))
randexp(rng::AbstractRNG, ::Sampler{Posit64}) = setprecision(BigFloat, 60) do; Posit64(randexp(rng, BigFloat)) end
randexp(rng::AbstractRNG, ::Sampler{LogPosit64}) = LogPosit64(randexp(rng, Float64)) # TODO

# miscellaneous
Base.bswap(t::AnyPosit) = Base.bswap_int(t)

# TODO: muladd?, rem?, mod?, random, isinf alias of isfinite?, posit?

end
