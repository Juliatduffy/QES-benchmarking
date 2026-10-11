# 3D figure: streamlines through the city, buildings in gray, colored by wind speed.
#
# GUI:   ParaView -> Edit > Reset Session -> View > Python Shell -> Run Script (this file)
# Batch: pvbatch --force-offscreen-rendering streamlines_3d.py   (see make_figures.sbatch)
from paraview.simple import *

DATA = '/scratch/general/vast/u1417379/claude-scratch/demo/okc/okc_rb32_windsOut.nc'
PNG = '/uufs/chpc.utah.edu/common/home/u1417379/QES/Testing/figures/streamlines_3d.png'
SEED_X = 40                  # m; just inside the 30 m empty border on the upwind (west) side
SEED_HEIGHTS = [5, 15, 30, 50]   # m above ground; one seed line per height
SEEDS_PER_LINE = 90
SPEED_RANGE = [0, 10]        # m/s; the 40 m lines reach ~9.5 m/s (2D figure at 4.5 m uses 0-7)

# --- data -------------------------------------------------------------------
reader = NetCDFReader(registrationName='okc', FileName=[DATA])
reader.Dimensions = '(z, y, x)'
reader.SphericalCoordinates = 0

vel = Calculator(registrationName='velocity', Input=reader)
vel.ResultArrayName = 'velocity'
vel.Function = 'u*iHat + v*jHat + w*kHat'

tubes = []
for z in SEED_HEIGHTS:
    st = StreamTracer(registrationName=f'streamlines_z{z}', Input=vel, SeedType='Line')
    st.Vectors = ['POINTS', 'velocity']
    st.SeedType.Point1 = [SEED_X, 40, z]
    st.SeedType.Point2 = [SEED_X, 1210, z]
    st.SeedType.Resolution = SEEDS_PER_LINE
    st.MaximumStreamlineLength = 3000
    tb = Tube(registrationName=f'tubes_z{z}', Input=st)
    tb.Radius = 0.7
    tubes.append(tb)

bld = Threshold(registrationName='buildings', Input=reader)
bld.Scalars = ['POINTS', 'icell']
bld.LowerThreshold = 0
bld.UpperThreshold = 0

ground = Plane(registrationName='ground')
ground.Origin = [0, 0, 0]
ground.Point1 = [1220, 0, 0]
ground.Point2 = [0, 1250, 0]

# --- display ----------------------------------------------------------------
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1800, 1100]
for tb in tubes:
    td = Show(tb, view)
    ColorBy(td, ('POINTS', 'mag'))
td.SetScalarBarVisibility(view, True)
# set the color range AFTER Show/ColorBy -- they rescale the color map to the data range
lut = GetColorTransferFunction('mag')
lut.ApplyPreset('Viridis (matplotlib)', True)
lut.AutomaticRescaleRangeMode = 'Never'
lut.RescaleTransferFunction(*SPEED_RANGE)
GetOpacityTransferFunction('mag').RescaleTransferFunction(*SPEED_RANGE)

bd = Show(bld, view)
bd.ColorArrayName = ['POINTS', '']
bd.DiffuseColor = bd.AmbientColor = [0.62, 0.62, 0.62]

gd = Show(ground, view)
gd.ColorArrayName = ['POINTS', '']
gd.DiffuseColor = gd.AmbientColor = [0.93, 0.93, 0.93]

sb = GetScalarBar(lut, view)
sb.Title = 'wind speed (m/s)'
sb.ComponentTitle = ''
sb.TitleColor = sb.LabelColor = [0, 0, 0]
sb.TitleFontSize = sb.LabelFontSize = 18
sb.DrawBackground = 1                     # white box behind the legend so lines/buildings don't show through
sb.BackgroundColor = [1, 1, 1, 1]         # RGBA, fully opaque
sb.BackgroundPadding = 8

title = Text(registrationName='title')
title.Text = 'QES-Winds, Oklahoma City test case: streamlines seeded on the west edge at z = 5, 15, 30 and 50 m'
tt = Show(title, view)
tt.WindowLocation = 'Upper Center'
tt.FontSize = 22
tt.Color = [0, 0, 0]

view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0

Render(view)   # ParaView resets the camera on the FIRST render, so render once before setting ours
# oblique view from the south-west, looking toward downtown
view.CameraFocalPoint = [620, 620, 10]
view.CameraPosition = [180, 60, 360]
view.CameraViewUp = [0, 0, 1]
view.CameraViewAngle = 30
Render(view)

SaveScreenshot(PNG, view, ImageResolution=[1800, 1100])
print('saved', PNG)
