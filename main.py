import itk
import vtk

print(itk.__version__)
print(vtk.__version__)

PixelType = itk.ctype("signed short")
FPixelType = itk.ctype("double")
Dimension = 3
    
ImageType = itk.Image[PixelType, Dimension]
FImageType = itk.Image[FPixelType, Dimension]

def get_image_reader(path):
    
    reader = itk.ImageFileReader[ImageType].New()
    
    reader.SetFileName(path)
    
    reader.Update()
    
    nrrdIo = itk.NrrdImageIO.New()
    
    
    reader.SetImageIO(nrrdIo)
    
    return reader

def char_to_float(img):
    castImageFilter = itk.CastImageFilter[ImageType, FImageType].New(Input=img)
    castImageFilter.Update()

    
    rescaler = itk.RescaleIntensityImageFilter[FImageType, FImageType].New()
    rescaler.SetInput(castImageFilter)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(1)
    rescaler.Update()
    return rescaler.GetOutput()

def float_to_char(img, max_v):
    rescaler = itk.RescaleIntensityImageFilter[FImageType, FImageType].New(Input=img)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(max_v)
    rescaler.Update()
    castImageFilter = itk.CastImageFilter[FImageType, ImageType].New(Input=rescaler.GetOutput())
    castImageFilter.Update()
    return castImageFilter.GetOutput()

img = get_image_reader('Data/case6_gre1.nrrd').GetOutput()


def do_translation_registration(fixed_image, moving_image,
        learningRate=4,
        minimumStepLength=0.001,
        relaxationFactor=0.5,
        numberOfIterations=200):
    
    transform_T = itk.TranslationTransform[itk.ctype("double"), Dimension]
    optimizer_T = itk.RegularStepGradientDescentOptimizerv4
    metric_T = itk.MeanSquaresImageToImageMetricv4[FImageType, FImageType]
    registration_T = itk.ImageRegistrationMethodv4[FImageType,FImageType]
    optimizer_T = itk.RegularStepGradientDescentOptimizerv4
    resample_T = itk.ResampleImageFilter[FImageType, FImageType]
    comp_transform_T = itk.CompositeTransform[itk.ctype("double"), Dimension]
    resample_T = itk.ResampleImageFilter
    interpolator_T = itk.WindowedSincInterpolateImageFunction[FImageType,3,itk.itkWindowedSincInterpolateImageFunctionPython.itkHammingWindowFunction3]
    caster_T = itk.CastImageFilter[FImageType, FImageType]
    
    initialTransform = transform_T.New()
    
    optimizer = optimizer_T.New()
    optimizer.SetLearningRate(learningRate)
    optimizer.SetMinimumStepLength(minimumStepLength)
    optimizer.SetRelaxationFactor(relaxationFactor)
    optimizer.SetNumberOfIterations(numberOfIterations)
    
    metric = metric_T.New()
    
    registration = registration_T.New()
    registration.SetFixedImage(fixed_image)
    registration.SetMovingImage(moving_image)
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInitialTransform(initialTransform)

    movingInitialTransform = transform_T.New()
    initialParameters = movingInitialTransform.GetParameters()
    initialParameters[0] = 0
    initialParameters[1] = 0
    movingInitialTransform.SetParameters(initialParameters)
    registration.SetMovingInitialTransform(movingInitialTransform)
    
    identityTransform = transform_T.New()
    identityTransform.SetIdentity()
    registration.SetFixedInitialTransform(identityTransform)
    
    registration.SetNumberOfLevels(1)
    registration.SetSmoothingSigmasPerLevel([0])
    registration.SetShrinkFactorsPerLevel([1])
    
    registration.Update()
    
    transform = registration.GetTransform()

    resample = resample_T.New(Input=fixed_image)
    resample.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
    resample.SetDefaultPixelValue(0.0)
    resample.SetInterpolator(interpolator_T.New())
    resample.SetTransform(transform)
    
    outputCompositeTransform = comp_transform_T.New()
    outputCompositeTransform.AddTransform(movingInitialTransform)
    outputCompositeTransform.AddTransform(registration.GetModifiableTransform())
    
    resampler = resample_T.New(Input=moving)
    resampler.SetTransform(outputCompositeTransform)
    resampler.SetUseReferenceImage(True)
    resampler.SetReferenceImage(fixed)
    resampler.SetDefaultPixelValue(0)
    
    caster = caster_T.New(Input=resampler)
    
    caster.Update()
    return caster.GetOutput()
fixed = char_to_float(get_image_reader('Data/case6_gre1.nrrd').GetOutput())
moving = char_to_float(get_image_reader('Data/case6_gre2.nrrd').GetOutput())
registrated = do_translation_registration(fixed, moving)

img_1 = fixed
img_2 = registrated

def watershed_labelization(input_img, min_size=200):
    diffuser_T = itk.GradientAnisotropicDiffusionImageFilter[FImageType, FImageType]
    gradient_T = itk.GradientMagnitudeImageFilter
    watershed_T = itk.WatershedImageFilter
    relabel_T = itk.RelabelComponentImageFilter[itk.Image[itk.ULL,Dimension],ImageType]

    diffuser = diffuser_T.New(Input=input_img)
    diffuser.SetNumberOfIterations(30)
    diffuser.SetTimeStep(0.04)
    diffuser.SetConductanceParameter(3)
    diffuser.Update()
    
    
    gradient_magnitude = gradient_T.New(Input=diffuser.GetOutput())
    gradient_magnitude.Update()
    
    watershed = watershed_T.New(Input=gradient_magnitude.GetOutput())
    watershed.SetThreshold(0.01)
    watershed.SetLevel(0.2)
    watershed.Update()
    
    relabel = relabel_T.New(Input=watershed.GetOutput())
    relabel.SetMinimumObjectSize(min_size)
    relabel.Update()
    return relabel.GetOutput()


def display_image(itk_image, name):
    rescaler = itk.RescaleIntensityImageFilter[FImageType, FImageType].New(Input=itk_image)
    rescaler.SetOutputMinimum(0)
    rescaler.SetOutputMaximum(255)
    rescaler.Update()
    vtk_image = itk.vtk_image_from_image(rescaler.GetOutput())

    

    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(20, 0.0)
    opacity_transfer_function.AddPoint(255, 0.3)
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    color_transfer_function.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
    color_transfer_function.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
    color_transfer_function.AddRGBPoint(255.0, 0.0, 0.2, 0.0)
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)

    volume_mapper = vtk.vtkSmartVolumeMapper()
    volume_mapper.SetInputData(vtk_image)
    
    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    interactor =  vtk.vtkRenderWindowInteractor()
    renderer.AddActor(volume)
    interactor.SetRenderWindow(window)
    renderer.SetBackground(0.2, 0.2, 0.2)
    renderer.GetActiveCamera().Azimuth(90)
    renderer.GetActiveCamera().Elevation(270)
    renderer.ResetCameraClippingRange() 
    renderer.ResetCamera()
    window.SetSize(500, 500)
    window.SetWindowName(name)
    window.Render()
    interactor.Initialize()
    interactor.Start()

def getselector(segmented, segment_id):
    return itk.BinaryThresholdImageFilter.New(
            Input=segmented,
            LowerThreshold=segment_id,
            UpperThreshold=segment_id,
            InsideValue=1,
            OutsideValue=0
    )

def selector_union(left, right):
    res = itk.OrImageFilter.New(Input1=left,Input2=right)
    res.Update()
    return res.GetOutput()
import functools   
def multiple_segments(original, segmented, segment_list):
    selectors = [getselector(segmented, segment_id) for segment_id in segment_list]
    selector = functools.reduce(selector_union, selectors)
    mask = itk.MaskImageFilter.New(
        Input=original,
        MaskImage=selector
    )
    mask.Update()
    highlighted = mask.GetOutput()
    
    origin_rescaler = itk.RescaleIntensityImageFilter[FImageType, FImageType].New(Input=original)
    origin_rescaler.SetOutputMinimum(0)
    origin_rescaler.SetOutputMaximum(30)
    origin_rescaler.Update()

    highlight_rescaler = itk.RescaleIntensityImageFilter[FImageType, FImageType].New(Input=highlighted)
    highlight_rescaler.SetOutputMinimum(0)
    highlight_rescaler.SetOutputMaximum(100)
    highlight_rescaler.Update()
    res = itk.AddImageFilter.New(
        Input1=origin_rescaler.GetOutput(),
        Input2=highlight_rescaler.GetOutput()
    )
    res.Update()
    return res.GetOutput()

segmented_1 = watershed_labelization(img_1)
segmented_2 = watershed_labelization(img_2)

display_image(multiple_segments(img_1, segmented_1, [2,3,4]), "tumor in scan 1")
display_image(multiple_segments(img_2, segmented_2, [3,4,8]), "tumor in scan 2")