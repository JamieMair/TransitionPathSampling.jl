using TPS
using Documenter

DocMeta.setdocmeta!(TPS, :DocTestSetup, :(using TPS); recursive=true)

makedocs(;
    modules=[TPS],
    authors="Jamie Mair <jamie.mair@hotmail.co.uk> and contributors",
    repo="https://github.com/JamieMair/TPS.jl/blob/{commit}{path}#{line}",
    sitename="TPS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JamieMair.github.io/TPS.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JamieMair/TPS.jl",
    devbranch="main",
)
