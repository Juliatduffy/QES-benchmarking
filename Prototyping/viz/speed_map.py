# 2D figure: wind speed on a horizontal slice, buildings in gray, arrows for direction.
#
# GUI:   ParaView -> Edit > Reset Session -> View > Python Shell -> Run Script (this file)
# Batch: pvbatch --force-offscreen-rendering speed_map.py   (see make_figures.sbatch)
from paraview.simple import *

DATA = '/scratch/general/vast/u1417379/claude-scratch/demo/okc/okc_rb32_windsOut.nc'
PNG = '/uufs/chpc.utah.edu/common/home/u1417379/QES/Testing/figures/speed_map.png'
Z_SLICE = 4.5          # m; cell centers are at 1.5, 4.5, 7.5, ... so this hits one layer exactly
SPEED_RANGE = [0, 7]   # m/s; fixed so every figure uses the same color scale

# --- data -------------------------------------------------------------------
reader = NetCDFReader(registrationName='okc', FileName=[DATA])
reader.Dimensions = '(z, y, x)'
reader.SphericalCoordinates = 0

vel = Calculator(registrationName='velocity', Input=reader)
vel.ResultArrayName = 'velocity'
vel.Function = 'u*iHat + v*jHat + w*kHat'

cut = Slice(registrationName='slice_z', Input=vel)
cut.SliceType = 'Plane'
cut.SliceType.Origin = [0, 0, Z_SLICE]
cut.SliceType.Normal = [0, 0, 1]

arrows = Glyph(registrationName='arrows', Input=cut, GlyphType='Arrow')
arrows.OrientationArray = ['POINTS', 'velocity']
arrows.ScaleArray = ['POINTS', 'No scale array']    # same length everywhere: arrows show direction only
arrows.ScaleFactor = 32
arrows.GlyphMode = 'Uniform Spatial Distribution (Bounds Based)'
arrows.MaximumNumberOfSamplePoints = 450

bld = Threshold(registrationName='buildings', Input=reader)
bld.Scalars = ['POINTS', 'icell']                  # icell 0 = solid building cell
bld.LowerThreshold = 0
bld.UpperThreshold = 0

# --- display ----------------------------------------------------------------
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1500, 1300]

sd = Show(cut, view)
ColorBy(sd, ('POINTS', 'mag'))
lut = GetColorTransferFunction('mag')          # set the range AFTER ColorBy, which rescales it
lut.ApplyPreset('Viridis (matplotlib)', True)
lut.AutomaticRescaleRangeMode = 'Never'
lut.RescaleTransferFunction(*SPEED_RANGE)
GetOpacityTransferFunction('mag').RescaleTransferFunction(*SPEED_RANGE)

ad = Show(arrows, view)
ad.ColorArrayName = ['POINTS', '']
ad.DiffuseColor = ad.AmbientColor = [0.1, 0.1, 0.1]

bd = Show(bld, view)
bd.ColorArrayName = ['POINTS', '']
bd.DiffuseColor = bd.AmbientColor = [0.6, 0.6, 0.6]

sb = GetScalarBar(lut, view)
sd.SetScalarBarVisibility(view, True)
sb.Title = 'wind speed (m/s)'
sb.ComponentTitle = ''
sb.TitleColor = sb.LabelColor = [0, 0, 0]
sb.TitleFontSize = sb.LabelFontSize = 18

title = Text(registrationName='title')
title.Text = f'QES-Winds, Oklahoma City test case: wind speed at z = {Z_SLICE} m (wind from the west)'
td = Show(title, view)
td.WindowLocation = 'Upper Center'
td.FontSize = 22
td.Color = [0, 0, 0]

view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 0
view.AxesGrid.Visibility = 1
view.AxesGrid.XTitle, view.AxesGrid.YTitle = 'x (m)', 'y (m)'
for p in ('XTitleColor', 'YTitleColor', 'XLabelColor', 'YLabelColor', 'GridColor'):
    setattr(view.AxesGrid, p, [0, 0, 0])
view.AxesGrid.XTitleFontSize = view.AxesGrid.YTitleFontSize = 18
view.AxesGrid.XLabelFontSize = view.AxesGrid.YLabelFontSize = 16

Render(view)   # ParaView resets the camera on the FIRST render, so render once before setting ours
# straight-down, orthographic camera (like a map)
view.CameraParallelProjection = 1
view.CameraFocalPoint = [610, 625, 0]
view.CameraPosition = [610, 625, 3000]
view.CameraViewUp = [0, 1, 0]
view.ResetCamera(False)       # refit along our view direction; also recomputes the clipping range
view.CameraParallelScale = 760
Render(view)

SaveScreenshot(PNG, view, ImageResolution=[1500, 1300])
print('saved', PNG)
