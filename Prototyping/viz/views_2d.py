# 2D slice figures in "streamplot" style, rendered with VTK through ParaView:
#   - top view at z = Z_TOP and side view through the tallest building
#   - wind speed as color (viridis), buildings grey, white evenly spaced in-plane streamlines with arrowheads
#
# How it works: the 3D QES grid is read with the NetCDF reader, one plane is pulled out with numpy, and that
# plane is rebuilt as a flat 2D vtkImageData (TrivialProducer) so VTK's vtkEvenlySpacedStreamlines2D
# ("Evenly Spaced Streamlines 2D") can trace streamlines in it. The side view puts height z on the screen's
# vertical axis.
#
# Batch: pvbatch --force-offscreen-rendering views_2d.py   (see make_figures.sbatch)
import numpy as np
from paraview.simple import *
from paraview import servermanager
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

DATA = '/scratch/general/vast/u1417379/claude-scratch/demo/okc/okc_rb32_windsOut.nc'
FIG = '/uufs/chpc.utah.edu/common/home/u1417379/QES/Testing/figures'
CASE = 'Oklahoma City test case (QES-Winds, solver only)'
Z_TOP = 4.5                 # m, height of the top view (a cell-center height: 1.5, 4.5, 7.5, ...)
TOP_RANGE = [0, 7]          # m/s color range, top view
SIDE_RANGE = [0, 10]        # m/s color range, side view (faster air above the roofs)
SIDE_HALF_WIDTH = 450       # m of x shown on each side of the tallest building

# --- read the 3D grid once and get numpy arrays shaped (nz, ny, nx) -------------------------------------
reader = NetCDFReader(registrationName='okc', FileName=[DATA])
reader.Dimensions = '(z, y, x)'
reader.SphericalCoordinates = 0
grid = servermanager.Fetch(reader)
nx, ny, nz = grid.GetDimensions()
(ox, oy, oz), (dx, dy, dz) = grid.GetOrigin(), grid.GetSpacing()
x, y, z = ox + dx * np.arange(nx), oy + dy * np.arange(ny), oz + dz * np.arange(nz)   # cell centers


def field(name):
    return vtk_to_numpy(grid.GetPointData().GetArray(name)).reshape(nz, ny, nx).astype(np.float64)


u, v, w, icell = field('u'), field('v'), field('w'), field('icell')
solid = icell == 0
speed = np.sqrt(u ** 2 + v ** 2 + w ** 2)
print(f'grid {nx}x{ny}x{nz}, origin ({ox}, {oy}, {oz}), spacing ({dx}, {dy}, {dz})')


def image_source(name, a_centers, b_centers, arrays, cells, height):
    """Flat 2D vtkImageData in screen coords (a -> screen x, b -> screen y) at screen z = height.
    cells=True: one cell per data value (crisp, like pcolormesh). cells=False: one point per value."""
    na, nb = len(a_centers), len(b_centers)
    da, db = a_centers[1] - a_centers[0], b_centers[1] - b_centers[0]
    img = vtkImageData()
    if cells:
        img.SetDimensions(na + 1, nb + 1, 1)
        img.SetOrigin(a_centers[0] - da / 2, b_centers[0] - db / 2, height)
    else:
        img.SetDimensions(na, nb, 1)
        img.SetOrigin(a_centers[0], b_centers[0], height)
    img.SetSpacing(da, db, 1)
    data = img.GetCellData() if cells else img.GetPointData()
    for key, arr in arrays.items():                  # arr shaped (nb, na), or (nb, na, 3) for vectors
        flat = np.ascontiguousarray(arr).reshape(na * nb, -1) if arr.ndim == 3 else np.ascontiguousarray(arr).ravel()
        vtk_arr = numpy_to_vtk(flat, deep=1)
        vtk_arr.SetName(key)
        data.AddArray(vtk_arr)
    src = TrivialProducer(registrationName=name)
    src.GetClientSideObject().SetOutput(img)
    src.UpdatePipeline()
    return src


def open_air_point(a, b, bld, ia, ib):
    """(a, b) of the nearest non-building point to index (ia, ib), searching along b."""
    for off in range(len(b)):
        for jb in (ib + off, ib - off):
            if 0 <= jb < len(b) and not bld[jb, ia]:
                return a[ia], b[jb]
    raise RuntimeError('no open-air point found')


def draw(tag, a, b, spd, va, vb, bld, rng, separation, arrow_size, stride, title, xlabel, ylabel, png, view_size,
         extra_line=None):
    """spd, va, vb, bld: 2D arrays shaped (len(b), len(a))."""
    color = image_source(f'{tag}_speed', a, b, {f'{tag}_speed': np.where(bld, np.nan, spd)}, cells=True, height=0)
    vec = np.stack([np.where(bld, 0, va), np.where(bld, 0, vb), np.zeros_like(va)], axis=-1)
    flow = image_source(f'{tag}_vel', a, b, {'vel2d': vec}, cells=False, height=20)   # above the colors

    lines = EvenlySpacedStreamlines2D(registrationName=f'{tag}_streamlines', Input=flow)
    lines.Vectors = ['POINTS', 'vel2d']
    # the filter grows every streamline outward from the first one, so the seed must be in open air
    # (a seed inside a building has zero velocity and produces nothing): use the upwind (west) edge
    lines.StartPosition = [*open_air_point(a, b, bld, 3, len(b) // 2), 20]
    lines.SeparatingDistance = separation           # in cell lengths
    lines.MaximumSteps = 20000
    heads = Glyph(registrationName=f'{tag}_arrowheads', Input=lines, GlyphType='Cone')
    heads.OrientationArray = ['POINTS', 'vel2d']
    heads.ScaleArray = ['POINTS', 'No scale array']
    heads.GlyphType.Radius = 0.3
    heads.ScaleFactor = arrow_size                  # arrowhead length in m
    heads.GlyphMode = 'Every Nth Point'
    heads.Stride = stride                           # one arrowhead every `stride` streamline points

    view = CreateView('RenderView')
    view.ViewSize = view_size
    cd = Show(color, view)
    ColorBy(cd, ('CELLS', f'{tag}_speed'))
    lut = GetColorTransferFunction(f'{tag}_speed')
    lut.ApplyPreset('Viridis (matplotlib)', True)
    lut.AutomaticRescaleRangeMode = 'Never'
    lut.RescaleTransferFunction(*rng)
    lut.NanColor = [0.5, 0.5, 0.5]                  # building cells are NaN -> grey
    cd.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title, sb.ComponentTitle = 'wind speed (m/s),  grey = building', ''
    sb.TitleColor = sb.LabelColor = [0, 0, 0]
    sb.TitleFontSize = sb.LabelFontSize = 16

    for src in (lines, heads):
        d = Show(src, view)
        d.ColorArrayName = ['POINTS', '']
        d.DiffuseColor = d.AmbientColor = [1, 1, 1]
    GetDisplayProperties(lines, view).LineWidth = 1.3
    hd = GetDisplayProperties(heads, view)
    hd.Ambient, hd.Diffuse = 1.0, 0.0              # flat white, no 3D shading
    if extra_line is not None:                      # e.g. where the side view cuts the top view
        ln = Line(registrationName=f'{tag}_cut', Point1=extra_line[0], Point2=extra_line[1])
        ld = Show(ln, view)
        ld.ColorArrayName = ['POINTS', '']
        ld.DiffuseColor = ld.AmbientColor = [0.85, 0.1, 0.1]
        ld.LineWidth = 2.5

    t = Text(registrationName=f'{tag}_title')
    t.Text = title
    td = Show(t, view)
    td.WindowLocation = 'Upper Center'
    td.FontSize = 20
    td.Color = [0, 0, 0]

    view.UseColorPaletteForBackground = 0
    view.Background = [1, 1, 1]
    view.OrientationAxesVisibility = 0
    ag = view.AxesGrid
    ag.Visibility = 1
    ag.XTitle, ag.YTitle = xlabel, ylabel
    for p in ('XTitleColor', 'YTitleColor', 'XLabelColor', 'YLabelColor', 'GridColor'):
        setattr(ag, p, [0, 0, 0])
    ag.XTitleFontSize = ag.YTitleFontSize = 18
    ag.XLabelFontSize = ag.YLabelFontSize = 15

    # 2D camera: look straight down -z, fit the plane with a margin for titles/axes
    Render(view)                                    # the first render resets the camera
    a0, a1 = a[0] - (a[1] - a[0]) / 2, a[-1] + (a[1] - a[0]) / 2
    b0, b1 = b[0] - (b[1] - b[0]) / 2, b[-1] + (b[1] - b[0]) / 2
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = [(a0 + a1) / 2, (b0 + b1) / 2, 0]
    view.CameraPosition = [(a0 + a1) / 2, (b0 + b1) / 2, 5000]
    view.CameraViewUp = [0, 1, 0]
    view.ResetCamera(False)                         # recompute clipping range
    aspect = view_size[0] / view_size[1]
    view.CameraParallelScale = max((b1 - b0) / 2 * 1.35, (a1 - a0) / 2 * 1.3 / aspect)
    view.CameraFocalPoint = [(a0 + a1) / 2 + 0.04 * (a1 - a0), (b0 + b1) / 2, 0]   # nudge left of the color bar
    view.CameraPosition = [(a0 + a1) / 2 + 0.04 * (a1 - a0), (b0 + b1) / 2, 5000]
    Render(view)
    SaveScreenshot(png, view, ImageResolution=view_size)
    print('saved', png)


# --- side view location: the row (y) through the tallest building --------------------------------------
top_k = np.where(solid.any(axis=0), nz - 1 - np.argmax(solid[::-1], axis=0), -1)   # highest solid k per (j, i)
height = np.where(top_k >= 0, z[np.clip(top_k, 0, None)] + dz / 2, 0.0)
j_tall, i_tall = np.unravel_index(np.argmax(height), height.shape)
y_side, x_tall, h_tall = y[j_tall], x[i_tall], height[j_tall, i_tall]
print(f'tallest building: {h_tall:.1f} m at x = {x_tall:.0f}, y = {y_side:.0f}')

# --- top view ----------------------------------------------------------------------------------------------
k = int(np.argmin(np.abs(z - Z_TOP)))
draw('top', x, y, speed[k], u[k], v[k], solid[k], TOP_RANGE, separation=2.5, arrow_size=16, stride=70,
     title=f'{CASE}\ntop view at z = {z[k]:.1f} m   (red line = side view cut)',
     xlabel='x (m)  (east)', ylabel='y (m)  (north)',
     png=f'{FIG}/top_view.png', view_size=[1500, 1350],
     extra_line=([x[0], y_side, 40], [x[-1], y_side, 40]))

# --- side view ---------------------------------------------------------------------------------------------
i0 = int(np.searchsorted(x, x_tall - SIDE_HALF_WIDTH))
i1 = int(np.searchsorted(x, x_tall + SIDE_HALF_WIDTH))
k1 = int(min(nz, np.searchsorted(z, 1.8 * h_tall) + 1))
xs, zs = x[i0:i1], z[:k1]
sl = (slice(0, k1), j_tall, slice(i0, i1))            # arrays (z, x) at this y
draw('side', xs, zs, speed[sl], u[sl], w[sl], solid[sl], SIDE_RANGE, separation=1.3, arrow_size=8, stride=90,
     title=f'{CASE}\nside view at y = {y_side:.0f} m (through the tallest building)',
     xlabel='x (m)  (east)', ylabel='height z (m)',
     png=f'{FIG}/side_view.png', view_size=[1900, 800])
