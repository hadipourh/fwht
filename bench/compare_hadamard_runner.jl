#!/usr/bin/env julia
# Run Hadamard.jl FWHT on a binary input vector and emit timing metadata.

using Hadamard
using Printf
using FFTW
using Base.Threads

function read_vector(path::AbstractString, n::Int)
    data = Vector{Float64}(undef, n)
    open(path, "r") do io
        read!(io, data)
    end
    return data
end

function write_vector(path::AbstractString, data::Vector{Float64})
    open(path, "w") do io
        write(io, data)
    end
end

function average_runtime!(buf::Vector{Float64}, input::Vector{Float64}, repeat::Int)
    total = 0.0
    for _ in 1:repeat
        buf .= input
        start = time_ns()
        fwht_natural!(buf)
        total += (time_ns() - start)
    end
    return (total / repeat) * 1e-9
end

function main()
    if length(ARGS) < 4
        println(stderr, "Usage: compare_hadamard_runner.jl <input.bin> <output.bin> <size> <repeat> [--scale-output]")
        exit(1)
    end

    FFTW.set_num_threads(max(1, nthreads()))

    input_path = ARGS[1]
    output_path = ARGS[2]
    n = parse(Int, ARGS[3])
    repeat = parse(Int, ARGS[4])
    scale_output = any(arg -> arg == "--scale-output", ARGS[5:end])

    input = read_vector(input_path, n)
    buf = copy(input)

    avg_time = average_runtime!(buf, input, repeat)

    if scale_output
        @. buf *= n
    end

    write_vector(output_path, buf)

    @printf("{\"time\":%.9f,\"repeat\":%d}\n", avg_time, repeat)
end

main()
