import vtk
import vtkmodules
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
import scipy, scipy.interpolate
import pyvista as pv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import SimpleITK as sitk
from scipy.optimize import curve_fit

def vtkNRRDreadear(path):
    mask_reader = vtk.vtkNrrdReader()
    mask_reader.SetFileName(path)
    mask_reader.Update()
    return mask_reader.GetOutput()

def GetModelFromLabelMask(image, numlabels, rangevalues):
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputData(image)
    dmc.ComputeNormalsOn()
    dmc.ComputeGradientsOn()
    dmc.GenerateValues(numlabels, rangevalues)      #rangevalues: range of labels
    dmc.Update()
    return dmc.GetOutput()

def GetModelFromImage(image, rangevalues):
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.ComputeNormalsOn()
    mc.GenerateValues(1, rangevalues)
    mc.Update()
    return mc.GetOutput()

def PolyDataSmoother(pd, iterations, featureangle, passband):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(pd)
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(featureangle)
    smoother.SetPassBand(passband)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()

def extractPolyDataBasedOnCriterion(pd):
    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(pd)
    connect.ScalarConnectivityOn()
    connect.Update()
    return connect.GetOutput()

def extractUnstructuredGridBasedOnCriterion(pd, lowerTh, upperTh):
    thr = vtk.vtkThreshold()
    thr.SetInputData(pd)
    thr.ThresholdByLower(lowerTh)
    thr.ThresholdByUpper(upperTh)
    return thr.Update()

def clipPolyDataBasedOnPlane(pd, direction):
    points = vtk_to_numpy(pd.GetPoints().GetData())
    points = np.reshape(points, (np.shape(points)[0], -1))
    points = np.transpose(points)
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    n = np.linalg.svd(M)[0][:, -2] * direction

    plane = vtk.vtkPlane()
    plane.SetOrigin(ctr)
    plane.SetNormal(n)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(pd)
    clipper.SetClipFunction(plane)
    clipper.SetValue(0)
    clipper.Update()
    return clipper.GetOutput()

def fitPlane(pd):
    points = vtk_to_numpy(pd.GetPoints().GetData())
    points = np.reshape(points, (np.shape(points)[0], -1))
    points = np.transpose(points)
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    n = np.linalg.svd(M)[0][:, -1]

    bestFitPlane = pv.Plane(center=ctr, direction=n, i_size=np.max(points[0])-np.min(points[0]), j_size=np.max(points[1])-np.min(points[1]))
    return bestFitPlane, n, ctr

def interpolatePlane(pd, factor):
    coordinates = vtk_to_numpy(pd.GetPoints().GetData())
    n_points = coordinates.shape[0]
    n_sample = round(n_points * factor)
    randomRow = np.random.default_rng().choice(np.arange(0, n_points), n_sample, replace=False)

    x = np.round(coordinates.T[0][randomRow], 3)
    y = np.round(coordinates.T[1][randomRow], 3)
    z = np.round(coordinates.T[2][randomRow], 3)

    spline = scipy.interpolate.Rbf(x, y, z, function='thin_plate', smooth=5, episilon=5)

    B1, B2 = np.meshgrid(x, y, indexing='xy')
    Z = spline(B1, B2)

    plane = pv.StructuredGrid(B1, B2, Z)

    return plane

def distanceFromPolyData(pd, point):
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(pd)

    return np.linalg.norm(distance.EvaluateFunction(point))

def graphGeodesicPath(pd, start, end):
    path = vtk.vtkDijkstraGraphGeodesicPath()
    path.SetInputData(pd)
    path.SetStartVertex(start)
    path.SetEndVertex(end)
    path.Update()
    return path.GetOutput()

def distanceBtwPolydata(pd1, pd2):
    clean1 = vtk.vtkCleanPolyData()
    clean1.SetInputData(pd1)
    clean2 = vtk.vtkCleanPolyData()
    clean2.SetInputData(pd2)
    distance = vtk.vtkDistancePolyDataFilter()
    distance.SetInputConnection(0, clean1.GetOutputPort())
    distance.SetInputConnection(1, clean2.GetOutputPort())
    distance.Update()
    return distance.GetOutput()

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly


def polynomial_regression3d(points):

    def func(x, a, b, c, d, e, f):
        return a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5 + f

    xdata = points.T[0]
    ydata = points.T[1]
    zdata = points.T[2]
    popt, _ = curve_fit(func, xdata, ydata)  # curve fit
    popt2, _ = curve_fit(func, xdata, zdata)
    x_line = np.arange(min(xdata), max(xdata), 0.5)
    a, b, c, d, e, f = popt  # calculate the output for the range
    y_line = func(x_line, a, b, c, d, e, f)
    a, b, c, d, e, f = popt2
    z_line = func(x_line, a, b, c, d, e, f)
    points = np.array([x_line, y_line, z_line]).T
    line = lines_from_points(points)
    return line

def interpolateAnnulus(pd):

    filter = sitk.BinaryDilateImageFilter()
    filter.SetBackgroundValue(0)
    filter.SetForegroundValue(1)
    filter.SetKernelType(sitk.sitkBSpline)
    filter.SetKernelRadius([1, 1, 1])
    dilateLabel = filter.Execute(annulus)

    dilateLabel_a = sitk.GetArrayFromImage(dilateLabel)
    dilateLabel_pd = pv.PolyData(dilateLabel_a)

def calc_mean_normal(pd):
    normalsFilter = vtk.vtkPolyDataNormals()
    normalsFilter.SetInputData(pd)
    normalsFilter.ComputePointNormalsOff()
    normalsFilter.ComputeCellNormalsOn()
    normalsFilter.Update()
    mean_normal = np.mean(vtk_to_numpy(normalsFilter.GetOutput().GetCellData().GetArray('Normals')), 0)
    return mean_normal

def calc_tangent_to_point(pd):
    points = pd.points
    tangent = points[1:] - points[:-1]
    tangent = np.vstack((tangent, [0,0,0]))
    return tangent

def posePlane(coaptation_line, plane, id):
    points = vtk_to_numpy(plane.GetPoints().GetData())
    points = np.reshape(points, (np.shape(points)[0], -1))
    points = np.transpose(points)
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    n = np.linalg.svd(M)[0][:, -1]

    tangent = calc_tangent_to_point(coaptation_line)[id]

    # finding norm of the vector n
    n_norm = np.sqrt(sum(n ** 2))

    proj_of_u_on_n = (np.dot(tangent, n) / n_norm ** 2) * n

    direction = tangent - proj_of_u_on_n
    normal = direction / np.sqrt(sum(direction ** 2))

    points = vtk_to_numpy(coaptation_line.GetPoints().GetData())
    points = np.reshape(points, (np.shape(points)[0], -1))
    points = np.transpose(points)

    bestFitPlane = pv.Plane(center=coaptation_line.points[id], direction=direction,
                            i_size=0.5*(np.max(points[0])-np.min(points[0])),
                            j_size=np.max(points[1])-np.min(points[1]))
    return bestFitPlane, normal