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
"poly_14", "LayerName:=", "Trace", "lw:=", "0mm", "n:=", 200, "U:=", "mm", x:=165.0, y:=-165.0, x:=165.0025869946004, y:=-164.99683597915026, x:=165.00903278156468, y:=-164.98637407026274, x:=165.01716149736052, y:=-164.967465831848, x:=165.02458269189646, y:=-164.9394045800153, x:=165.02876630159452, y:=-164.9019580165563, x:=165.0271220021628, y:=-164.85538941109766, x:=165.0170813128198, y:=-164.80046679995212, x:=164.99618076252585, y:=-164.73845988443622, x:=164.96214439621443, y:=-164.67112453805274, x:=164.9129638957244, y:=-164.6006750614443, x:=164.84697461630498, y:=-164.52974455274392, x:=164.76292589494273, y:=-164.46133398520485, x:=164.66004407063673, y:=-164.39875080016708, x:=164.53808676798712, y:=-164.345538027999, x:=164.39738713249733, y:=-164.3053951392733, x:=164.23888686685166, y:=-164.2820919999519, x:=164.06415709978265, y:=-164.2793774548616, x:=163.87540632029012, y:=-164.3008841906375, x:=163.67547482693578, y:=-164.35003163033255, x:=163.4678153714451, y:=-164.42992868514534, x:=163.2564599144343, y:=-164.54327823271214, x:=163.04597265509076, y:=-164.6922852050826, x:=162.84138974231104, y:=-164.87857015223196, x:=162.64814631830237, y:=-165.10309009860006, x:=162.47199178315117, y:=-165.36606843099315, x:=162.3188943965471, y:=-165.66693544700934, x:=162.19493654704164, y:=-166.00428105518642, x:=162.10620221636347, y:=-166.37582095298924, x:=162.0586583430871, y:=-166.77837741865946, x:=162.0580319432728, y:=-167.20787564033773, x:=162.10968497279998, y:=-167.6593562736104, x:=162.2184890145789, y:=-168.1270046699239, x:=162.38870194160884, y:=-168.60419695664535, x:=162.62384874234442, y:=-169.0835628786546, x:=162.926608696863, y:=-169.55706503516254, x:=163.29871106020616, y:=-170.0160938680251, x:=163.7408413427818, y:=-170.45157748331263, x:=164.25256017715753, y:=-170.85410512047855, x:=164.8322366267286, y:=-171.21406282729083, x:=165.47699762589912, y:=-171.52177965779944, x:=166.18269504533245, y:=-171.76768248891415, x:=166.94389165174496, y:=-171.94245735235734, x:=167.7538669823144, y:=-172.0372150062711, x:=168.60464388213512, y:=-172.0436583277165, x:=169.48703616274528, y:=-171.95424899646548, x:=170.39071753436943, y:=-171.76237086420923, x:=171.30431164824617, y:=-171.46248736349065, x:=172.2155027625604, y:=-171.05029030875275, x:=173.11116622056343, y:=-170.52283747879295, x:=173.9775176070555, y:=-169.878676446026, x:=174.8002791341866, y:=-169.11795223313672, x:=175.56486150417163, y:=-168.2424965312512, x:=176.2565592095973, y:=-167.25589640442976, x:=176.86075696598604, y:=-166.16354063129648, x:=177.3631447304311, y:=-164.9726420936562, x:=177.7499385484296, y:=-163.69223491119658, x:=178.00810429220286, y:=-162.33314533752895, x:=178.1255812111111, y:=-160.907935772162, x:=178.09150211113675, y:=-159.43082160139295, x:=177.89640691823112, y:=-157.91756095406387, x:=177.5324463615077, y:=-156.38531784089156, x:=176.99357253816584, y:=-154.85249953363203, x:=176.2757131934271, y:=-153.33856942749375, x:=175.3769266658431, y:=-151.86383701168342, x:=174.2975346106576, y:=-150.4492269434147, x:=173.04022982043432, y:=-149.1160295748392, x:=171.61015671123204, y:=-147.88563561496358, x:=170.0149623319585, y:=-146.77925791467368, x:=168.26481608130712, y:=-145.81764363770478, x:=166.37239667746974, y:=-145.0207803193, x:=164.35284531668015, y:=-144.40759951328292, x:=162.22368437315734, y:=-143.99568188366464, x:=160.0047014303178, y:=-143.80096770553286, x:=157.71779888597294, y:=-143.83747679919506, x:=155.38680983705257, y:=-144.11704192932342, x:=153.037281416378, y:=-144.6490596557503, x:=150.69622721913538, y:=-145.44026252382903, x:=148.39185091386264, y:=-146.4945163298146, x:=146.15324357579, y:=-147.81264599114834, x:=144.01005770317164, y:=-149.3922932941506, x:=141.99216127380288, y:=-151.22780948445185, x:=140.12927556346736, y:=-153.310185311212, x:=138.45060077507057, y:=-155.6270207381555, x:=136.9844338115559, y:=-158.1625360966734, x:=135.75778276262386, y:=-160.89762598332044, x:=134.79598286056918, y:=-163.80995670109243, x:=134.1223187905354, y:=-166.87410751657598, x:=133.75765831211626, y:=-170.06175545947318, x:=133.72010216012637, y:=-173.3419028335985, x:=134.0246551408478, y:=-176.68114604594066, x:=134.68292322521927, y:=-180.04398379977397, x:=135.702841262118, y:=-183.39316214615562, x:=137.08843569374784, y:=-186.6900533526389, x:=138.83962635262466, y:=-189.89506503573622, x:=140.9520710580066, y:=-192.96807552159183, x:=143.41705631186582, y:=-195.86889095419681, x:=146.2214369244599, y:=-198.55771926877412, x:=149.34762688175115, y:=-200.99565579571274, x:=152.7736432065842, y:=-203.1451749632384, x:=156.4732039685206, y:=-204.9706223298756, x:=160.41588097001377, y:=-206.43870100510017, x:=164.56730698613814, y:=-207.51894641208392, x:=168.88943676879816, y:=-208.18418331305818, x:=173.340860351974, y:=-208.41095905769575, x:=177.87716652018787, y:=-208.17994712933287, x:=182.45135363617945, y:=-207.476315253233, x:=187.01428437409945, y:=-206.29005259493204, x:=191.51518027961492, y:=-204.61625091360045, x:=195.9021514863703, y:=-202.45533494296956, x:=200.12275636717334, y:=-199.8132377474746, x:=204.12458539573962, y:=-196.7015173397239, x:=207.85586304800034, y:=-193.13741144225298, x:=211.26606118758883, y:=-189.14382792594623, x:=214.3065170642186, y:=-184.74926915298457, x:=216.9310488116862, y:=-179.9876891864255, x:=219.0965611687669, y:=-174.8982835937072, x:=220.76363406513968, y:=-169.5252123590442, x:=221.8970867185406, y:=-163.91725722102592, x:=222.4665099805492, y:=-158.12741555748542, x:=222.44675984768634, y:=-152.21243374045483, x:=221.81840532175238, y:=-146.23228367010637, x:=220.56812415742652, y:=-140.24958695841013, x:=218.68904047388037, y:=-134.32899196120584, x:=216.18099872729945, y:=-128.53650954218557, x:=213.0507691384703, y:=-122.93881408481133, x:=209.31218033871798, y:=-117.60251683987971, x:=204.98617573222452, y:=-112.59341919915596, x:=200.1007908660026, y:=-107.97575391183483, x:=194.69104994255179, y:=-103.81142260379644, x:=188.798780495782, y:=-100.1592382138859, x:=182.4723461687123, y:=-97.07418112176533, x:=175.7662984718129, y:=-94.60667780435142, x:=168.7409493531568, y:=-92.80191081953502, x:=161.4618673650403, y:=-91.69916877503347, x:=153.99930115532544, y:=-91.33124469621187, x:=146.42753493436481, y:=-91.72389086011134, x:=138.82418145879348, y:=-92.8953377155038, x:=131.26941892079623, y:=-94.8558839635365, x:=123.84517892490092, y:=-97.6075642346122, x:=116.63429346367806, y:=-101.14390006990405, x:=109.71960945909981, y:=-105.44973910681689, x:=103.18308000868441, y:=-110.5011864842993, x:=97.1048419565189, y:=-116.26563153477102, x:=91.5622897914166, y:=-122.70187182400187, x:=86.62915615130615, y:=-129.76033554892607, x:=82.37460937912783, y:=-137.383402217102, x:=78.86237862676157, y:=-145.50582042206128, x:=76.14991693687784, y:=-154.05522040823973, x:=74.28761254635462, y:=-162.95271800015072, x:=73.31805834870252, y:=-172.1136053656428, x:=73.27538902775845, y:=-181.44812300540502, x:=74.18469483313851, y:=-190.8623063230669, x:=76.06152031331825, y:=-200.25889914494792, x:=78.91145555984298, y:=-209.53832563790868, x:=82.7298266524952, y:=-218.59971122965266, x:=87.50149103795107, y:=-227.3419423792447, x:=93.20074253247918, y:=-235.66475438694346, x:=99.79132952258483, y:=-243.46983588101438, x:=107.22658875730457, y:=-250.66193818344425, x:=115.44969589407017, y:=-257.14997744356697, x:=124.39403268956764, y:=-262.84811724456563, x:=133.9836694312522, y:=-267.67681933721053, x:=144.13395989825418, y:=-271.5638502412867, x:=154.75224483668785, y:=-274.44523167964485, x:=165.73865864857152, y:=-276.26612317292063, x:=176.9870327403877, y:=-276.98162562332845, x:=188.38588777141766, y:=-276.55749535065917, x:=199.8195058977507, y:=-274.97075880819716, x:=211.16907303926467, y:=-272.2102190947221, x:=222.31388021726127, y:=-268.2768463835504, x:=233.13257213244447, y:=-263.18404550175336, x:=243.50443038826597, y:=-256.9577951019423, x:=253.3106781239853, y:=-249.6366541637441, x:=262.43579231458796, y:=-241.27163292955427, x:=270.76880962911474, y:=-231.92592680554188, x:=278.2046115217173, y:=-221.6745132295007, x:=284.64517416605565, y:=-210.60361300651923, x:=290.00076893707853, y:=-198.8100191255373, x:=294.19109939667413, y:=-186.40029757816916, x:=297.1463611513134, y:=-173.48986618897453, x:=298.8082115190364, y:=-160.20195891683795, x:=299.1306366665807, y:=-146.66648448358274, x:=298.0807047499821, y:=-133.01878951201206, x:=295.6391946067118, y:=-119.39833759534073, x:=291.80109069577725, y:=-105.94731685823291, x:=286.57593625400887, y:=-92.80918959203234, x:=279.9880380202395, y:=-80.12719843992531, x:=272.0765173610852, y:=-68.04284435957061, x:=262.89520419806, y:=-56.69435219037416, x:=252.5123717701448, y:=-46.215140090775805, x:=241.01031195198743, y:=-36.73230937996497, x:=228.48475256810073, y:=-28.365171412398258, x:=215.0441198795282, y:=-21.223828028219387, x:=200.80865115276507, y:=-15.40782185594777, x:=185.90936393226394, y:=-11.004872295326109, x:=170.48689030855553, y:=-8.089712379707859, x:=154.69018608492394, y:=-6.723040912534408, x:=138.6751262780953, y:=-6.950603297002953, x:=122.60299982446173, y:=-8.802413339629833))
 oProject.Save