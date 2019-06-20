module Arrays

export Array

const Buffer{T} = Base.Vector{T}

Buffer{T}(len::Int) where {T} = Buffer{T}(undef, len)
Buffer(ptr::Ptr{T}, len::Int; own::Bool=false) where {T} = Base.unsafe_wrap(Base.Array, ptr, len, own=own)
# unsafe_copyto!(::Buffer{T}, doffs, ::Buffer{T}, soffs, n)

# Array
function _prod(dims::NTuple{N, Int}) where {N}
    x = 1
    for i = 1:N
        @inbounds x *= dims[i]
    end
    return x
end

struct Array{T, N} <: AbstractArray{T, N}
    data::Buffer{T}
    dims::NTuple{N, Int}
    offset::UInt
    length::UInt

    Array{T, N}(data::Buffer{T}, dims::NTuple{N, Int}, o::Integer, l::Integer) where {T, N} =
        new{T, N}(data, dims, UInt(o), UInt(l))
    function Array{T, N}(dims::NTuple{N, Int}) where {T, N}
        len = _prod(dims)
        data = Buffer{T}(len)
        A = new{T, N}(data, dims, 0, len)
        return A
    end
end

# constructors
# type and dimensionality specified, accepting dims as series of Ints
Array{T, 1}(::UndefInitializer, m::Int) where {T} = Array{T, 1}((m,))
Array{T, 2}(::UndefInitializer, m::Int, n::Int) where {T} = Array{T, 2}((m, n))
Array{T, 3}(::UndefInitializer, m::Int, n::Int, o::Int) where {T} = Array{T, 3}((m, n, o))
Array{T, N}(::UndefInitializer, d::Vararg{Int, N}) where {T, N} = Array{T, N}(d)

# type and dimensionality specified, accepting dims as tuples of Ints
Array{T, 1}(::UndefInitializer, d::NTuple{1, Int}) where {T} = Array{T, 1}(undef, getfield(d, 1))
Array{T, 2}(::UndefInitializer, d::NTuple{2, Int}) where {T} = Array{T, 2}(undef, getfield(d, 1), getfield(d, 2))
Array{T, 3}(::UndefInitializer, d::NTuple{3, Int}) where {T} = Array{T, 3}(undef, getfield(d, 1), getfield(d, 2), getfield(d, 3))
Array{T, N}(::UndefInitializer, d::NTuple{N, Int}) where {T, N} = Array(T, d)

# type but not dimensionality specified
Array{T}(::UndefInitializer, m::Int) where {T} = Array{T, 1}(undef, m)
Array{T}(::UndefInitializer, m::Int, n::Int) where {T} = Array{T, 2}(undef, m, n)
Array{T}(::UndefInitializer, m::Int, n::Int, o::Int) where {T} = Array{T, 3}(undef, m, n, o)
Array{T}(::UndefInitializer, d::NTuple{N, Int}) where {T, N} = Array{T, N}(undef, d)

# empty vector constructor
Array{T, 1}() where {T} = Array{T, 1}(undef, 0)

(::Type{Array{T, N} where T})(x::AbstractArray{S, N}) where {S, N} = Array{S, N}(x)

Array(A::AbstractArray{T, N})    where {T, N}   = Array{T, N}(A)
Array{T}(A::AbstractArray{S, N}) where {T, N, S} = Array{T, N}(A)

# array.jl
const Vector{T} = Array{T, 1}
const Matrix{T} = Array{T, 2}
const VecOrMat{T} = Union{Vector{T}, Matrix{T}}

vect() = Vector{Any}()
vect(X::T...) where {T} = T[ X[i] for i = 1:length(X) ]

function vect(X...)
    T = Base.promote_typeof(X...)
    return copyto!(Vector{T}(undef, length(X)), X)
end

Base.size(A::Array) = A.dims
Base.IndexStyle(::Type{<:Array}) = Base.IndexLinear()

elsize(::Type{<:Array{T}}) where {T} = isbitstype(T) ? sizeof(T) : (Base.isbitsunion(T) ? Base.bitsunionsize(T) : sizeof(Ptr))
Base.sizeof(a::Array) = Core.sizeof(a.data)

Base.isassigned(A::Array, i::Int...) = isassigned(A.data, i...)
Base.getindex(A::Vector, i::Int) = A.data[A.offset + i]
Base.getindex(A::Array, i::Int) = A.data[i]
Base.setindex!(A::Vector, x, i::Int) = setindex!(A.data, x, i)
Base.setindex!(A::Array, x, i::Int) = setindex!(A.data, x, i)

Base.pointer(A::Vector) = pointer(A.data) + A.offset
Base.pointer(A::Array) = pointer(A.data)

Base.unsafe_copyto!(dest::Array{T}, doffs, src::Array{T}, soffs, n) where T =
    unsafe_copyto!(dest.data, doffs + dest.offset, src.data, soffs + src.offset, n)

function Base.copyto!(dest::Array{T}, doffs::Integer, src::Array{T}, soffs::Integer, n::Integer) where T
    n == 0 && return dest
    n > 0 || Base._throw_argerror()
    if soffs < 1 || doffs < 1 || soffs+n-1 > length(src) || doffs+n-1 > length(dest)
        throw(BoundsError())
    end
    unsafe_copyto!(dest, doffs, src, soffs, n)
    return dest
end

Base.copyto!(dest::Array{T}, src::Array{T}) where {T} = copyto!(dest, 1, src, 1, length(src))

Base.copy(A::Array{T, N}) where {T, N} = Array{T, N}(copy(A.data), A.dims, A.offset, A.length)

## Constructors ##

Base.similar(a::Array{T,1}) where {T}                    = Vector{T}(undef, size(a,1))
Base.similar(a::Array{T,2}) where {T}                    = Matrix{T}(undef, size(a,1), size(a,2))
Base.similar(a::Array{T,1}, S::Type) where {T}           = Vector{S}(undef, size(a,1))
Base.similar(a::Array{T,2}, S::Type) where {T}           = Matrix{S}(undef, size(a,1), size(a,2))
Base.similar(a::Array{T}, m::Int) where {T}              = Vector{T}(undef, m)
Base.similar(a::Array, T::Type, dims::Dims{N}) where {N} = Array{T,N}(undef, dims)
Base.similar(a::Array{T}, dims::Dims{N}) where {T,N}     = Array{T,N}(undef, dims)

function Base.getindex(::Type{T}, vals...) where T
    a = Vector{T}(undef, length(vals))
    @inbounds for i = 1:length(vals)
        a[i] = vals[i]
    end
    return a
end

Base.getindex(::Type{T}) where {T} = (Base.@_inline_meta; Vector{T}())
Base.getindex(::Type{T}, x) where {T} = (Base.@_inline_meta; a = Vector{T}(undef, 1); @inbounds a[1] = x; a)
Base.getindex(::Type{T}, x, y) where {T} = (Base.@_inline_meta; a = Vector{T}(undef, 2); @inbounds (a[1] = x; a[2] = y); a)
Base.getindex(::Type{T}, x, y, z) where {T} = (Base.@_inline_meta; a = Vector{T}(undef, 3); @inbounds (a[1] = x; a[2] = y; a[3] = z); a)

function Base.getindex(::Type{Any}, @nospecialize vals...)
    a = Vector{Any}(undef, length(vals))
    @inbounds for i = 1:length(vals)
        a[i] = vals[i]
    end
    return a
end
Base.getindex(::Type{Any}) = Vector{Any}()

function unsafe_wrap(::Union{Type{Array}, Type{Array{T}}, Type{Array{T, N}}},
                          ptr::Ptr{T}, dims::NTuple{N, Int}; own::Bool=false) where {T, N}
    len = _prod(dims)
    data = Buffer(ptr, len; own=own)
    return Array{T, N}(data, dims, 0, len)
end

unsafe_wrap(T::Type, p::Ptr, len::Int; own::Bool=false) = unsafe_wrap(T, p, (len,); own=own)
unsafe_wrap(T::Type, p::Ptr, dims::NTuple{N,<:Integer}; own::Bool=false) where {N} =
    unsafe_wrap(T, p, convert(Tuple{Vararg{Int}}, dims), own=own)

function unsafe_wrap(::Type{Vector{UInt8}}, s::String)
    len = sizeof(s)
    data = Buffer(pointer(s), len; own=true)
    return Vector{UInt8}(data, (len,), 0, len)
end

Base.String(v::Vector{UInt8}) = unsafe_string(pointer(v), length(v))

end # module
