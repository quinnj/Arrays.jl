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

const ALIGNMENT = 64
align(x, sz) = (x + sz - 1) & -sz

zero!(ptr, n) = ccall(:memset, Cvoid, (Ptr{Cvoid}, Cint, Csize_t), ptr, 0, n)

mutable struct Buffer{T} <: AbstractVector{T}
    ptr::Ptr{T}
    len::Int

    function Buffer(::Type{T}, nelements::Int) where {T}
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
        len = align(len, ALIGNMENT)
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

function Base.getindex(b::Buffer{T}, i::Int) where {T}
    @boundscheck checkbounds(b, i)
    if isbitstype(T)
        val = GC.@preserve b unsafe_load(b.ptr, i)
        return val
    elseif isbitsunion(T)
        sel = unsafe_load(convert(Ptr{UInt8}, b.ptr + bitsunionsize(T) * b.len), i)
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
        GC.@preserve b unsafe_store!(convert(Ptr{UInt8}, b.ptr + bitsunionsize(T) * b.len), sel, i)
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

struct Array{T, N} <: AbstractArray{T, N}
    data::Buffer{T}
    dims::NTuple{N, Int}
    offset::UInt
    length::UInt

    function Array(::Type{T}, dims::NTuple{N, Int}) where {T, N}
        len = 0
        for i = 1:N
            len *= dims[i]
        end
        data = Buffer(T, len)
        A = new{T, N}(data, dims, 0, len)
        return A
    end
end


end # module
