# notice the units resolve themselves.
∂x_∂Δy′(Δy′::Meters, Δy::Meters, H::Meters; f=25.0mm) = 
    1/2 * -f * Δy / (Δy′)^2 |> upreferred


∂x_∂Δx′(Δx′::Meters, Δx::Meters, H::Meters; f=25.0mm) =
    -1/2 * (f*H*Δx / Δx′^2) / sqrt((Δx/2)^2 + f*H*Δx/Δx′)
