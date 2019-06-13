module Arrays

export Array

export arraysize, arraylen, arrayref, arrayset

Base.@pure isbitsunion(u) = u isa Union ? ccall(:jl_array_store_unboxed, Cint, (Any,), u) != Cint(0) : false

Base.@pure function bitsunionsize(u::Union)
    sz = Ref{Csize_t}(0)
    algn = Ref{Csize_t}(0)
    isunboxed = ccall(:jl_islayout_inline, Cint, (Any, Ptr{Csize_t}, Ptr{Csize_t}), u, sz, algn)
    @assert isunboxed != Cint(0)
    return sz[]
end

# isassigned
function typelayout(::Type{T}) where {T}
    sz = Ref{Csize_t}(0)
    algn = Ref{Csize_t}(0)
    isunboxed = ccall(:jl_islayout_inline, Cint, (Any, Ptr{Csize_t}, Ptr{Csize_t}), T, sz, algn)
    return isunboxed != Cint(0), sz, algn
end

Base.@pure isunion(u) = u isa Union

const LARGE_ALIGNMENT = 64
const SMALL_ALIGNMENT = 16
align(x, sz) = (x + sz - 1) & -sz

zero!(ptr, n) = ccall(:memset, Cvoid, (Ptr{Cvoid}, Cint, Csize_t), ptr, 0, n)

mutable struct Buffer{T} <: AbstractVector{T}
    ptr::Ptr{T}
    len::Int

    Buffer{T}(ptr::Ptr{T}, len::Int) where {T} = new{T}(ptr, len)
    function Buffer(::Type{T}, nelements::Int, alignment=LARGE_ALIGNMENT) where {T}
        isunboxed, sz, algn = typelayout(T)
        if isunboxed
            # data stored in region
            elsize = sizeof(T)
            len = elsize * nelements
            if isunion(T)
                # extra "selector" byte stored for each isbits union element
                len += nelements
            elseif elsize == 1
                # all byte buffers include extra null-terminating byte
                len += 1
            end
        else
            # only pointers to objects are stored
            len = sizeof(Ptr{Cvoid}) * nelements
        end
        len = align(len, alignment)
        ptr = convert(Ptr{T}, Libc.malloc(len))
        if !isunboxed || isunion(T)
            zero!(ptr, len)
        end
        if isunboxed && elsize == 1
            unsafe_store!(ptr, 0, len)
        end
        buffer = new{T}(ptr, nelements)
        finalizer(x->Libc.free(x.ptr), buffer)
        return buffer
    end
end

Base.size(b::Buffer) = (b.len,)

Base.checkbounds(b::Buffer, i::Int) = i <= b.len || throw(BoundsError(b, i))

Base.@pure nth_union(u, n) = u isa Union ? (n == 0 ? u.a : nth_union(u.b, n - 1)) : u

typetagptr(b::Buffer{T}) = convert(Ptr{UInt8}, b.ptr + bitsunionsize(T) * b.len)

function Base.getindex(b::Buffer{T}, i::Int) where {T}
    @boundscheck checkbounds(b, i)
    if isbitstype(T)
        val = GC.@preserve b unsafe_load(b.ptr, i)
        return val
    elseif isbitsunion(T)
        sel = unsafe_load(typetagptr(b), i)
        val = GC.@preserve b unsafe_load(convert(Ptr{nth_union(T, sel)}, b.ptr + bitsunionsize(T) * (i - 1)))
        return val
    else
        ptr = GC.@preserve b unsafe_load(convert(Ptr{Ptr{T}}, b.ptr), i)
        if ptr == C_NULL
            throw(UndefRefError())
        end
        val = GC.@preserve ptr unsafe_pointer_to_objref(ptr)
        return val
    end
end

Base.@pure find_union(u, ::Type{T}, i=0) where {T} = u isa Union ? (u.a === T ? i : find_union(u.b, T, i + 1)) : i

function Base.setindex!(b::Buffer{T}, x::S, i::Int) where {T, S}
    @boundscheck checkbounds(b, i)
    if isbitstype(T)
        GC.@preserve b unsafe_store!(b.ptr, x, i)
    elseif isbitsunion(T)
        GC.@preserve b unsafe_store!(convert(Ptr{S}, b.ptr + bitsunionsize(T) * (i - 1)), x)
        sel = find_union(T, S)
        GC.@preserve b unsafe_store!(typetagptr(b), sel, i)
    else
        # FIXME: how to tell the GC that the Buffer "owns" x now?
        GC.@preserve x unsafe_store!(convert(Ptr{Ptr{T}}, b.ptr), pointer_from_objref(x), i)
    end
    return b
end

function bufferunset(b::Buffer{T}, i::Int) where {T}
    @boundscheck checkbounds(b, i)
    if !isbitstype(T) && !isbitsunion(T)
        GC.@preserve b unsafe_store!(convert(Ptr{Ptr{Cvoid}}, b.ptr), C_NULL, i)
    end
    return b
end

function Base.isassigned(b::Buffer{T}, i::Int) where {T}
    if i < 0 || i > b.len
        return false
    elseif !isbitstype(T) && !isbitsunion(T)
        ptr = GC.@preserve b unsafe_load(convert(Ptr{Ptr{T}}, b.ptr), i)
        return ptr != C_NULL
    end
    return true
end

Base.pointer(b::Buffer) = b.ptr

function Base.copy(b::Buffer{T}) where {T}
    newb = Buffer(T, b.len)
    unsafe_copyto!(b.ptr, newb.ptr, b.len)
    if isbitsunion(T)
        unsafe_copyto!(typetagptr(b), typetagptr(newb), b.len)
    end
    return newb
end

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
    function Array(::Type{T}, dims::NTuple{N, Int}) where {T, N}
        len = _prod(dims)
        data = Buffer(T, len)
        A = new{T, N}(data, dims, 0, len)
        return A
    end
end

const Vector{T} = Array{T, 1}
const Matrix{T} = Array{T, 2}

Base.size(A::Array) = A.dims
Base.IndexStyle(::Type{<:Array}) = Base.IndexLinear()

Base.isassigned(A::Array, i::Int) = isassigned(A.data, i)
Base.getindex(A::Vector, i::Int) = A.data[A.offset + i]
Base.getindex(A::Array, i::Int) = A.data[i]
Base.setindex!(A::Vector, x, i::Int) = setindex!(A.data, x, i)
Base.setindex!(A::Array, x, i::Int) = setindex!(A.data, x, i)

# constructors
# type and dimensionality specified, accepting dims as series of Ints
Array{T, 1}(::UndefInitializer, m::Int) where {T} = Array(T, (m,))
Array{T, 2}(::UndefInitializer, m::Int, n::Int) where {T} = Array(T, (m, n))
Array{T, 3}(::UndefInitializer, m::Int, n::Int, o::Int) where {T} = Array(T, (m, n, o))
Array{T, N}(::UndefInitializer, d::Vararg{Int, N}) where {T, N} = Array(T, d)

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

Base.pointer(A::Vector) = pointer(A.data) + A.offset
Base.pointer(A::Array) = pointer(A.data)

function Base.unsafe_wrap(::Type{Vector{UInt8}}, s::String)
    len = sizeof(s)
    data = Buffer{UInt8}(pointer(s), len)
    return Vector{UInt8}(data, (len,), 0, len)
end

function Base.unsafe_wrap(::Union{Type{Array}, Type{Array{T}}, Type{Array{T, N}}},
                          ptr::Ptr{T}, dims::NTuple{N, Int}; own::Bool=false) where {T, N}
    len = _prod(dims)
    # TODO: handle own argument
    data = Buffer{T}(ptr, len)
    return Array{T, N}(data, dims, 0, len)
end
Base.unsafe_wrap(T::Type, p::Ptr, len::Int; own::Bool=false) = unsafe_wrap(T, p, (len,); own=own)
# Base.unsafe_wrap(T::Type, p::Ptr, dims::NTuple{N,<:Integer}; own::Bool=false) where {N} =
#     unsafe_wrap(T, p, convert(Tuple{Vararg{Int}}, dims), own=own)

Base.copy(A::Array{T, N}) where {T, N} = Array{T, N}(copy(A.data), A.dims, A.offset, A.length)

Base.String(v::Vector{UInt8}) = unsafe_string(pointer(v), length(v))

end # module
