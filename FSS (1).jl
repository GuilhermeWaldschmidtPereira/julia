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
    r = 0.0
    theta = 0
    theta_1 = 0
    theta2 = 0
    theta3 = 0
    theta4 = 0
    n = 2
    for i in 1:200
        x = (r*theta)*cos(theta_1+pi/4)/n #verde
        y = (r*theta)*sin(theta_1+pi/4)/n #verde
        x2 = (r*theta)*cos(theta_1+pi/1.4)/n #preto
        y2 = (r*theta)*sin(theta_1+pi/1.4)/n #preto
        x3 = (r*theta)*cos(theta_1-pi/1.3)/n #vermelho 
        y3 = (r*theta)*sin(theta_1-pi/1.3)/n #vermelho 
        x4 = (r*theta)*cos(theta_1-pi/3.75)/n #azivis
        y4 = (r*theta)*sin(theta_1-pi/3.75)/n #azivis
        push!(v_x, x)
        push!(v_y, y)
        push!(v_x2, x2)
        push!(v_y2, y2)        
        push!(v_x3, x3)
        push!(v_y3, y3)
        push!(v_x4, x4)
        push!(v_y4, y4)
        theta+=0.0122
        theta_1+=0.1
        theta2+=0.1
        theta3+=0.1
        theta4+=0.1
        r+=0.067
    end
    return v_x, v_y, v_x2, v_y2, v_x3, v_y3, v_x4, v_y4
end
vx, vy, vx2, vy2, vx3, vy3, vx4, vy4 = polar()

v2x = (vx.+16.5)
v2y = (vy.-16.5)
v2x2 = (vx2.+16.5)
v2y2 = (vy2.+16.5)
v2x3 = (vx3.- 16.5)
v2y3 = (vy3.+ 16.5)
v2x4 = (vx4.- 16.5)
v2y4 = (vy4.-16.5)

plot(v2x,v2y, label = "", color = "green")
plot!(v2x2, v2y2, label = "", color = "black")
plot!(v2x3, v2y3, label = "", color = "red")
plot!(v2x4, v2y4, label = "", color = "blue")

x, y = v2x[197:200], v2y[197:200]
data = DataFrame(x=x, y=y)
model = lm(@formula(y ~0+ x), data)
println(coef(model))
x = 0:0.01:(v2x[199]+v2x[200])/2
y = predict(model, DataFrame(x=x))
plot!(x,y, color = "black", label = "")

x,y = v2x2[198:200], v2y2[198:200]
data = DataFrame(x=x, y=y)
model = lm(@formula(y ~ 0+x), data)
println(coef(model))
x = 0:0.1:v2x2[200]
y = predict(model, DataFrame(x=x))
plot!(x,y, color = "black", label = "")

x,y = v2x3[197:200], v2y3[197:200]
data = DataFrame(x=x, y=y)
model = lm(@formula(y ~ 0+x), data)
println(coef(model))
x = v2x3[200]:0.1:0.1
y = predict(model, DataFrame(x=x))
plot!(x,y, color = "black", label = "")

x,y = v2x4[197:200], v2y4[197:200]
data = DataFrame(x=x, y=y)
model = lm(@formula(y ~ 0+x), data)
println(coef(model))
x = v2x4[200]:0.1:0.1
y = predict(model, DataFrame(x=x))
plot!(x,y, color = "black", label = "")
num_linhas = length(x)

# Lê o arquivo
line_data = """
' ----------------------------------------------
' Script Recorded by Ansoft Designer Version 3.5.0
' 12:58 AM  Sep 27, 2024
' ----------------------------------------------
Dim oAnsoftApp
Dim oDesktop
Dim oProject
Dim oDesign
Dim oEditor
Dim oModule
Set oAnsoftApp = CreateObject("AnsoftDesigner.DesignerScript")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.RestoreWindow
Set oProject = oDesktop.SetActiveProject("teste")
Set oDesign = oProject.SetActiveDesign("PlanarEM1")
Set oEditor = oDesign.SetActiveEditor("Layout")


oEditor.CreatePolygon Array("NAME:Contents", "polyGeometry:=", Array("Name:=", _
"poly_14", "LayerName:=", "Trace", "lw:=", "0mm", "n:=", $num_linhas, "U:=", "mm", """

# Grava as coordenadas x e y do arquivo no padrão aceito pelo script
line_points = []

for (col1, col2) in zip(x, y)
    push!(line_points, "\"x:=\" * string(float($col1) * 10) * \", \" * \"y:=\" * string(float($col2) * 10)")
end

line_data *= join(line_points, ", ") * "))"

# Escreve o arquivo de saída no formato .vbs
open("teste.vbs", "w") do output_file
    write(output_file, line_data)
    write(output_file, "\n oProject.Save")
end

