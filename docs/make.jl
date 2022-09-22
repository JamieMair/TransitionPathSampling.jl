using TransitionPathSampling
using Documenter

DocMeta.setdocmeta!(TransitionPathSampling, :DocTestSetup, :(using TransitionPathSampling); recursive=true)

makedocs(;
    modules=[TransitionPathSampling],
    authors="Jamie Mair <jamie.mair@hotmail.co.uk> and contributors",
    repo="https://github.com/JamieMair/TransitionPathSampling.jl/blob/{commit}{path}#{line}",
    sitename="TransitionPathSampling.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JamieMair.github.io/TransitionPathSampling.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JamieMair/TransitionPathSampling.jl",
    devbranch="main",
)
