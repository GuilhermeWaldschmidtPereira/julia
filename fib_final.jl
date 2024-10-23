using Plots
using GLM, DataFrames
function polar()
    v_x = []
    v_y = []
    v_x2 = []
    v_y2 = []
    v_x3 = []
    v_y3 = []
    v_x4 = []
    v_y4 = []
    a = 1.8
    b = 0.165
    n = 20
    r = 0.0
    theta = 0
    n = 2
    for i in 1:150 
        x = a*cos(theta)*exp(1)^(b*theta)#verde
        y = a*sin(theta)*exp(1)^(b*theta)
        push!(v_x, x)
        push!(v_y, y)
        x2 = a*cos(theta)*exp(1)^(b*theta)#preto
        y2 = a*sin(theta)*exp(1)^(b*theta)
        push!(v_x2, x2)
        push!(v_y2, y2)
        x3 = a*cos(theta+pi)*exp(1)^(b*theta)#vermelho
        y3 = a*sin(theta)*exp(1)^(b*theta)
        push!(v_x3, x3)
        push!(v_y3, y3)
        x4 = a*cos(theta+pi)*exp(1)^(b*theta)#azivis
        y4 = a*sin(theta)*exp(1)^(b*theta)
        push!(v_x4, x4)
        push!(v_y4, y4)
        theta+=0.1
    end
    (v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4) = (v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4)
    return v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4
end
vx, vy, vx2, vy2, vx3, vy3, vx4, vy4 = polar()
v2x = (vx.+20)
v2y = (vy.-20)
v2x2 = (vx2.+20)
v2y2 = (vy2.+12)
v2x3 = (vx3.- 20)
v2y3 = (vy3.+ 12)
v2x4 = (vx4.- 20)
v2y4 = (vy4.-20)
#plot(v2x,v2y, label = "", color = "green")
#plot!(v2x2, v2y2, label = "", color = "black")
#plot!(v2x3, v2y3, label = "", color = "red")
#plot!(v2x4, v2y4, label = "", color = "blue")

#a = (v2y2[1500]+v2y[1500])/2
#b = (0, a)
#scatter!(b)

x, y = v2x[149:150], v2y[149:150]
data = DataFrame(x=x, y=y)
data.x2 = data.x .^ 2
model = lm(@formula(y ~ 0+ x + x2), data)
println(coef(model))
x = 0.0:0.01:v2x[150]
y = predict(model, DataFrame(x=x, x2=x.^2))
#plot!(x,y, color = "black", label = "")
v = []
kp = []
append!(v,x)
append!(kp,y)
v = reverse(v)
kp = reverse(kp)
append!(v2x,v)
append!(v2y,kp)


x,y = v2x2[149:150], v2y2[149:150]
data = DataFrame(x=x, y=y)
data.x2 = data.x .^ 2
model = lm(@formula(y ~ x + x2+0), data)
println(coef(model))
x = 0:0.01:v2x2[150]
y = predict(model, DataFrame(x=x, x2=x.^2))
plot!(x,y, color = "blue", label = "")
v2 = []
kp2 = []
append!(v2,x)
append!(kp2,y)
v2 = reverse(v2)
kp2 = reverse(kp2)
append!(v2x2,v2)
append!(v2y2,kp2) 

x,y = v2x3[149:150], v2y3[149:150]
data = DataFrame(x=x, y=y)
data.x2 = data.x .^ 2
model = lm(@formula(y ~ x + x2+0), data)
println(coef(model))
x = v2x3[150]:0.01:0
y = predict(model, DataFrame(x=x, x2=x.^2))
plot!(x,y, color = "black", label = "")
append!(v2x3, x)  
append!(v2y3, y) 

x,y = v2x4[149:150], v2y4[149:150]
data = DataFrame(x=x, y=y)
data.x2 = data.x .^ 2
model = lm(@formula(y ~ x + x2+0), data)
println(coef(model))
x = v2x4[150]:0.01:0
y = predict(model, DataFrame(x=x, x2=x.^2))
plot!(x,y, label = "", color = "black")
append!(v2x4, x)  
append!(v2y4, y) 

using DelimitedFiles
matreg = hcat(v,kp)
mat = hcat(v2x,v2y)
mat2 = hcat(v2x2,v2y2)
mat3 = hcat(v2x3,v2y3)
mat4 = hcat(v2x4,v2y4)
writedlm("c:\\Users\\glaub\\OneDrive\\Área de Trabalho\\Projeto_Robson\\espirais\\espiraisfibv2\\fibv02_reg1.txt", mat)
writedlm("c:\\Users\\glaub\\OneDrive\\Área de Trabalho\\Projeto_Robson\\espirais\\espiraisfibv2\\fibv02_reg2.txt", mat2)
writedlm("c:\\Users\\glaub\\OneDrive\\Área de Trabalho\\Projeto_Robson\\espirais\\espiraisfibv2\\fibv02_reg3.txt", mat3)
writedlm("c:\\Users\\glaub\\OneDrive\\Área de Trabalho\\Projeto_Robson\\espirais\\espiraisfibv2\\fibv02_reg4.txt", mat4)
plot!(v2x, v2y, label = "")
#scatter!(v2x, v2y)
#scatter!(v2x2, v2y2)
plot!(v2x2, v2y2, label = "")
plot!(v2x3, v2y3, label = "")
plot!(v2x4, v2y4, label = "")
