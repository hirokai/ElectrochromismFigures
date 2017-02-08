using Plots
using Colors

plotly()

function plot_cv(path)
  name = splitdir(path)[end]
  rpm = match(r"\s(\S+)rpm",name).captures[1]
  max_count = 100000
  vss = zeros(max_count,2)
  csv = readlines(path)
  i = 1
  for line in csv[5422:8423]
#  for line in csv[22:end]
    # line2 = chomp(line)
    vs = split(line,r", ")
    if size(vs)[1] < 2
      break
    end
    vs2 = collect(float(v) for v in vs[1:2])
    vss[i,:] = vs2
    # println(line)
    i += 1
  end

  vss = vss[1:i-1,:]
  rpms = ["500","1k","2k","3k","4k","5k"]
  colors = linspace(colorant"green",colorant"red",size(rpms)[1])
  idx = findfirst(rpms .== rpm)
  color = idx > 0 ? colors[idx] : colorant"gray"
  println(rpm,"\t",idx)
  Plots.plot!(vss[:,1],vss[:,2],size=(1200,800),label=string(rpm,": ",name),c=color)
end

function main()
  files = ["61 30wt 500rpm.txt", "62 30wt 500rpm.txt", "71 30wt 1krpm.txt", "72 30wt 1krpm.txt", "74 30wt 2krpm.txt",
  "86 30wt 3krpm.txt", "87 30wt 3krpm.txt", "88 30wt 4krpm.txt", "89 30wt 4krpm.txt", "90 30wt 5krpm.txt",
  "91 30wt 5krpm.txt"]
  Plots.plot()
  for n = files
    plot_cv(joinpath("../data/cv/1021",n))
  end
  gui()
end

main()
