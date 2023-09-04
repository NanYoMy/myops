import numpy as np
# import tensorflow as tf
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk

def to_categorical(y, num_classes=None, dtype='float32'):

  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical

def padd(min,max,padding,size):
    start = 0 if min - padding < 0 else min - padding
    stop = (size - 1) if max + padding > (size - 1) else max + padding
    return slice(start, stop)

def get_bounding_boxV2(x,padding=0):
    res=[]
    coor = np.nonzero(x)
    size=np.shape(x)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res

def reindex_label(img, ids):
    arr = sitk.GetArrayFromImage(img)
    new_array = reindex_label_array(arr, ids)
    new_img = sitk.GetImageFromArray(new_array)
    new_img.CopyInformation(img)


def reindex_label_array(arr, ids):
    new_array = np.zeros(arr.shape, np.uint16)
    for k in ids.keys():
        for i in ids[k]:
            new_array = new_array + np.where(arr == i, k, 0)
    return new_array

def get_mask_bounding_box(x, padding=0, dim=3):
    res=[]
    # coor = np.nonzero(x)
    if isinstance(x,sitk.Image):
        x=sitk.GetArrayFromImage(x)
    coor = np.nonzero(x)
    size=np.shape(x)
    for i in range(dim):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res


def get_bounding_box_by_id(x,padding=0,id=5):
    res=[]
    # coor = np.nonzero(x)
    if id is not None:
        x=np.where(x==id,1,0)
    coor = np.nonzero(x)
    size=np.shape(x)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res

def get_bounding_box_by_ids(x,padding=0,ids=[200,1220,2221]):
    res=[]
    # coor = np.nonzero(x)
    if isinstance(x,sitk.Image):
        x=sitk.GetArrayFromImage(x)
    out = binarize_numpy_array( x,ids)
    coor = np.nonzero(out)
    size=np.shape(out)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding,size[i]))
    return res
def get_bounding_box_by_idsV2(x,padding=[0,15,15],ids=[200,1220,2221]):
    res=[]
    # coor = np.nonzero(x)
    if isinstance(x,sitk.Image):
        x=sitk.GetArrayFromImage(x)
    out = binarize_numpy_array( x,ids)
    coor = np.nonzero(out)
    size=np.shape(out)
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding[i],size[i]))
    return res

def get_bounding_box_by_idsV3(x,padding=[0,0.05,0.05],ids=[200,1220,2221]):
    res=[]
    # coor = np.nonzero(x)
    if isinstance(x,sitk.Image):
        x=sitk.GetArrayFromImage(x)
    out = binarize_numpy_array( x,ids)
    coor = np.nonzero(out)
    size=np.shape(out)
    padding=[int(i*j) for i,j in zip(size,padding)]
    for i in range(3):
        xmin = np.min(coor[i])
        xmax = np.max(coor[i])
        res.append(padd(xmin,xmax,padding[i],size[i]))
    return res

def reduce_z(x,bbox,condition=[200,600,500]):
    pass



def convertArrayToImg(array,para=None):
    img=sitk.GetImageFromArray(array)
    if para is not None:
        img.CopyInformation(para)
    return img



def binarize_img(x,ids ):

    array=sitk.GetArrayFromImage(x)
    out = np.zeros(array.shape, dtype=np.uint16)
    for L in ids:
        out = out + np.where(array == L, 1, 0)

    out_img = convertArrayToImg(out, x)

    return out_img

def generate_LV_mask(img, ids={1: [200], 2: [1220], 3: [2221],4:[500]}):
    arr = sitk.GetArrayFromImage(img)
    new_array = np.zeros(arr.shape, np.uint16)
    for k in ids.keys():
        for i in ids[k]:
            new_array = new_array + np.where(arr == i, 1, 0)
    new_img = sitk.GetImageFromArray(new_array)
    new_img.CopyInformation(img)
    return new_img

def binarize_numpy_array(array,ids=None,ignore=0,outindex=1 ):

    out = np.zeros(array.shape, dtype=np.uint16)
    if ids==None:
        ids=np.unique(array)
    for L in ids:
        if L==ignore:
            continue
        out = out + np.where(array == L, outindex, 0)

    return out


def zoom3Darray(img, new_size):
    scale=img.shape/new_size
    return zoom(img, scale)

def resize3DArray(img, new_size):
    return resize(img,new_size)

def sitkResize3D(img,new_size):

    reference_image = sitk.Image(new_size, img.GetPixelIDValue())
    reference_image.SetOrigin(img.GetOrigin())
    reference_image.SetDirection(img.GetDirection())
    reference_image.SetSpacing([sz * spc / nsz for nsz, sz, spc in zip(new_size, img.GetSize(), img.GetSpacing())])
    return sitk.Resample(img, reference_image)
    #sitk.Resample(sitk.SmoothingRecursiveGaussian(grid_image, 2.0)

def get_rotate_ref_img( data):
    dimension = data.GetDimension()


    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Select arbitrary number of pixels per dimension, smallest size that yields desired results
    # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
    # often result in non-isotropic pixel spacing.
    reference_size = [0] * dimension

    reference_spacing = [0] * dimension
    print(data.GetDirection())
    new_size=np.matmul(np.reshape(np.array(data.GetDirection()),[3,3]),np.array(data.GetSize()))
    reference_size[0]=int(abs(new_size[0]))
    reference_size[1]=int(abs(new_size[1]))
    reference_size[2]=int(abs(new_size[2]))

    new_space=np.matmul(np.reshape(np.array(data.GetDirection()),[3,3]),np.array(data.GetSpacing()))
    reference_spacing[0]=float(abs(new_space[0]))
    reference_spacing[1]=float(abs(new_space[1]))
    reference_spacing[2]=float(abs(new_space[2]))

    reference_image = sitk.Image(reference_size, data.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    return reference_image

def sitkResize(image, new_size, interpolator):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    new_spacing = [sz * spc / nsz for nsz, sz, spc in zip(new_size, image.GetSize(), image.GetSpacing())]
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = list(image.GetSpacing())
    new_size=[oz*os/nz for oz,os,nz in zip(orig_size,orig_spacing,new_spacing)]
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage

def sitkRespacing(image, interpolator, spacing):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    new_spacing = spacing
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = list(image.GetSpacing())
    new_size=[oz*os/nz for oz,os,nz in zip(orig_size,orig_spacing,new_spacing)]
    new_size = np.ceil(new_size).astype(np.int)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage

# def ndarrayResize3D(array,new_size,interpolator):
#     img=sitk.GetImageFromArray(array)
#     img=sitkResize3DV2(img,new_size,interpolator)
#     return sitk.GetArrayFromImage(img)

def crop_by_bbox(img,bbox):
    crop_img = img[bbox[2].start:bbox[2].stop+1,bbox[1].start:bbox[1].stop+1,bbox[0].start:bbox[0].stop+1]
    return crop_img

def reindex_label(label,Label_Index=[200,1220,2221,500, 600]):
    array = sitk.GetArrayFromImage(label)
    for i,L in enumerate(Label_Index):
        array = np.where(array ==L , i+1, array)
    array=to_categorical(array)
    new_label=sitk.GetImageFromArray(array)
    new_label.CopyInformation(label)#这个函数关键

    return new_label

def reindex_label_array(array,Label_Index=[200,1220,2221,500, 600]):

    for i,L in enumerate(Label_Index):
        array = np.where(array ==L , i+1, array)
    array=to_categorical(array)
    return array

def reindex_label_by_dict(img, ids):
    arr = sitk.GetArrayFromImage(img)
    new_array = reindex_label_array_by_dict(arr, ids)
    new_img = sitk.GetImageFromArray(new_array)
    new_img.CopyInformation(img)
    return new_img


def reindex_label_array_by_dict(arr, ids):
    new_array = np.zeros(arr.shape, np.uint16)
    for k in ids.keys():
        for i in ids[k]:
            new_array = new_array + np.where(arr == i, k, 0)
    return new_array


def extract_label_bitwise(arr,ids={1:[2,4]}):
    new_array = np.zeros(arr.shape, np.uint16)
    for k in ids.keys():
        for i in ids[k]:
            tmp=np.bitwise_and(arr,i)
            new_array =new_array+ np.where(tmp == i, k, 0)
    return new_array

def reverse_one_hot(array):
    last_dim=array.shape[-1]
    out_shape=array.shape[:-1]
    y = array.ravel()
    array=array.reshape(y.shape[0]//last_dim,last_dim)
    array=np.argmax(array,axis=1)
    array=array.reshape(out_shape)
    return array


def normalize_sitk_img_mask(img,mask):
    img_array=sitk.GetArrayFromImage(img).astype(np.float32)
    mask_array=sitk.GetArrayFromImage(mask)
    indices=np.where(mask_array>0)
    mean=img_array[indices].mean()
    std=img_array[indices].std()
    img_array[indices]=(img_array[indices]-mean)/std

    #其他的值保持为0
    indices = np.where(mask_array <=0)
    img_array[indices]=0

    return convertArrayToImg(img_array,img)

def zscore_img_arr_with_mask(img_array, mask_array):

    indices=np.where(mask_array>0)
    mean=img_array[indices].mean()
    std=img_array[indices].std()
    img_array[indices]=(img_array[indices]-mean)/std

    #其他的值保持为0
    indices = np.where(mask_array <=0)
    img_array[indices]=0

    return img_array

def zscore_img_arr_with_mask(img_array, mask_array):

    indices=np.where(mask_array>0)
    mean=img_array[indices].mean()
    std=img_array[indices].std()
    img_array[indices]=(img_array[indices]-mean)/std

    #其他的值保持为0
    indices = np.where(mask_array <=0)
    img_array[indices]=0

    return img_array


def rescale_0_1_img_arr_with_mask(img_array, mask_array,clip=[10,90]):

    indices=np.where(mask_array>0)
    pixel=img_array[indices]
    pixel=np.sort(pixel)
    clipa=int(clip[0]/100*len(pixel))
    clipb=int(clip[1]/100*len(pixel))
    pixel=pixel[clipa : clipb]

    max=np.max(pixel)
    min=np.min(pixel)
    img_array=(img_array-min)/(max-min)

    #其他的值保持为0
    indices = np.where(mask_array <=0)
    img_array[indices]=0
    img_array=np.clip(img_array,0,1)

    return img_array

def rescale_one_dir(pathes, is_image=True):
    for path in pathes:
        img=sitk.ReadImage(path)
        # img=sitk.RescaleIntensity(img)
        img=clipseScaleSitkImage(img)
        sitk.WriteImage(img,path)


def clipseScaleSitkImage(sitk_image,low=5, up=95):
    np_image = sitk.GetArrayFromImage(sitk_image)
    # threshold image between p10 and p98 then re-scale [0-255]
    p0 = np_image.min().astype('float')
    p10 = np.percentile(np_image, low)
    p99 = np.percentile(np_image, up)
    p100 = np_image.max().astype('float')
    # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0,p5,p10,p90,p98,p100))
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p10,
                                upper=p100,
                                outsideValue=p10)
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p0,
                                upper=p99,
                                outsideValue=p99)
    sitk_image = sitk.RescaleIntensity(sitk_image,
                                       outputMinimum=0,
                                       outputMaximum=255)
    return sitk_image


def scaleArray(np_image,output_min=9,output_max=255):
    min = np_image.min().astype('float')
    max = np_image.max().astype('float')
    return (np_image-min)/(max-min)*(output_max-output_min)

def clipseScaleSArray(np_image,low=5, up=95,ouputmin=0,outputmax=255):

    # threshold image between p10 and p98 then re-scale [0-255]
    p0 = np_image.min().astype('float')
    p10 = np.percentile(np_image, low)
    p99 = np.percentile(np_image, up)
    p100 = np_image.max().astype('float')
    # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0,p5,p10,p90,p98,p100))
    sitk_image=sitk.GetImageFromArray(np_image)
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p10,
                                upper=p100,
                                outsideValue=p10)
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p0,
                                upper=p99,
                                outsideValue=p99)
    sitk_image = sitk.RescaleIntensity(sitk_image,
                                       outputMinimum=ouputmin,
                                       outputMaximum=outputmax)

    return sitk.GetArrayFromImage(sitk_image)






# def clipScaleImage(name, low=5, up=95):
#     sitk_image = sitk.ReadImage(name, sitk.sitkFloat32)
#     return clipseScaleSitkImage(sitk_image,low,up)


def merge_dir(slices):

    arrays=[]
    for s in slices:
        arrays.append(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(s))))
    arrays=np.stack(arrays)
    return arrays





