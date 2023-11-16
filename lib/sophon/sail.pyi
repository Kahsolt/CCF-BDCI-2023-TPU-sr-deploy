import sys
import enum
from typing import Any
import numpy

class Format(enum.Enum):
    FORMAT_YUV420P = 0
    FORMAT_YUV422P = 1
    FORMAT_YUV444P = 2
    FORMAT_NV12 = 3
    FORMAT_NV21 = 4
    FORMAT_NV16 = 5
    FORMAT_NV61 = 6
    FORMAT_NV24 = 7
    FORMAT_RGB_PLANAR = 8
    FORMAT_BGR_PLANAR = 9
    FORMAT_RGB_PACKED = 10
    FORMAT_BGR_PACKED = 11
    FORMAT_RGBP_SEPARATE = 12
    FORMAT_BGRP_SEPARATE = 13
    FORMAT_GRAY = 14
    FORMAT_COMPRESSED = 15

class ImgDtype(enum.Enum):
    DATA_TYPE_EXT_FLOAT32 = 0
    DATA_TYPE_EXT_1N_BYTE = 1
    DATA_TYPE_EXT_4N_BYTE = 2
    DATA_TYPE_EXT_1N_BYTE_SIGNED = 3
    DATA_TYPE_EXT_4N_BYTE_SIGNED = 4

class Dtype(enum.Enum):
    BM_FLOAT32 = 0
    BM_INT8 = 2
    BM_UINT8 = 3
    BM_INT32 = 6
    BM_UINT32 = 7

class IOMode(enum.Enum):
    SYSI = 0
    SYSO = 1
    SYSIO = 2
    DEVIO = 3

class bmcv_resize_algorithm(enum.Enum):
    BMCV_INTER_NEAREST = 0
    BMCV_INTER_LINEAR = 1
    BMCV_INTER_BICUBIC = 2

def get_available_tpu_num() -> int:
    """
    Get the number of available TPUs.

    Returns
    -------
    Number of available TPUs.
    """
    pass

def set_print_flag(print_flag: bool) -> None:
    """ Print main process time use  """
    pass

def set_dump_io_flag(dump_io_flag: bool) -> None:
    """ Dump io data"""
    pass

def set_decoder_env(env_name: str, env_value: str) -> None:
    """ Set Decoder environment, must set befor Decoder Constructor, else use default values
    
    refcounted_frames, extra_frame_buffer_num, rtsp_transport, stimeout, rtsp_flags, buffer_size, max_delay, probesize, analyzeduration
    """
    pass

def base64_encode(handle: Handle, a: bytes) -> bytes:
    """ Encode a base64 string """
    pass

def base64_decode(handle: Handle, a: bytes) -> bytes:
    """ Decoder a base64 string """
    pass

def base64_encode_array(handle: Handle, a: numpy.ndarray) -> bytes:
    """ Encode a base64 string """
    pass

def base64_decode_asarray(handle: Handle, a: bytes) -> numpy.ndarray:
    """ Decoder a base64 string """
    pass

class Handle:
    def __init__(self, dev_id: int) -> Handle:
        """ 
        Constructor with device id.

        Parameters:
        ----------
        dev_id: int
           ID of TPU.
        """
        pass

    def get_device_id(self) -> int:
        """ Get device id of this handle. """
        pass

    def get_sn(self) -> str:
        """ Get Serial Number. """
        pass


class bm_image:
    def width(self) -> int: pass
    def height(self) -> int: pass
    def format(self) -> Format: pass
    def dtype(self) -> ImgDtype: pass
        
class BMImage:
    def __init__(self) -> BMImage: pass

    def __init__(self, handle: Handle, 
        h: int, w: int, 
        format: Format, 
        dtype:ImgDtype) -> BMImage: pass

    def width(self) -> int: pass

    def height(self) -> int: pass

    def format(self) -> int: pass

    def dtype(self) -> int: pass

    def data(self) -> bm_image: 
        """ Get inner bm_image  """
        pass

    def get_plane_num(self) -> int: pass

    def need_to_free(self) -> bool: pass

    def empty_check(self) -> int: pass

    def get_device_id(self) -> int: pass

    def asmat(self) -> numpy.ndarray[numpy.uint8]: pass


class Decoder:
    def __init__(self, file_path: str, compressed: bool =  True, 
        dev_id: int = 0) -> Decoder:
        """ 
        Decoder by VPU

        Parameters:
        ----------
        file_path : str
            Path or rtsp url to the video/image file.
        compressed : bool, optional
            Whether the format of decoded output is compressed NV12.
            Defaults is True.
        dev_id : int, optional
            ID of TPU. 
            Defaults to 0.
        """
       pass

    def is_opened(self) -> bool:
       """ Judge if the source is opened successfully. """
       pass
    
    def get_frame_shape(self) -> list[int]:
        """ Get frame shape in the Decoder.
       
        Returns
        -------
        list[int], [1, C, H, W]
        """
        pass

    def read(self, handle: Handle) -> BMImage: pass
    # def read(self, handle: Handle, image: BMImage) -> int: pass
    def read_(self, handle: Handle, image: bm_image) -> None: pass    
    def get_fps(self) -> float: pass
    def release(self) -> None: pass
    def reconnect(self) -> int: pass

class Tensor:
    def __init__(self, handle: Handle, shape: list[int], 
        dtype: Dtype = Dtype.BM_FLOAT32,
        own_sys_data: bool = False,
        own_dev_data: bool = False) -> Tensor: pass
        
    def __init__(self, handle: Handle, data: numpy.ndarray[Any,numpy.dtype[Any]], own_sys_data: bool = True) -> Tensor: 
        """
        Constructor allocates device memory of the tensor use numpy.ndarray, \n
        Input numpy must be C_CONTIGUOUS True

        Parameters:
        ----------
        handle: Handle
        data: numpy.ndarray
            dtype is float_ | uint8 | int32 | int_, C_CONTIGUOUS flag must be True.
        own_sys_data: bool, default is True.
            Indicator of whether own system memory, If false, the memory will be copied to device directly  
        """
        pass
  
    def shape(self) -> list[int]: pass

    def reshape(self, shape: list[int]) -> None: pass

    def own_sys_data(self) -> bool: pass

    def own_dev_data(self) -> bool: pass


    def asnumpy(self, shape: list[int] = None) -> numpy.ndarray[Any,numpy.dtype[Any]] : 
        """
        Get ndarray in system memory of the tensor.

        Parameters:
        ----------
        shape: list[int], optional
            If provided, Shape of output data. 
        """
        pass

    def update_data(self, data: numpy.ndarray[Any, numpy.dtype[numpy.any]]) -> None: 
        """
        If own_sys_data, Update system data of the tensor. \n
        Else if own_dev_data Update device data of the tensor. 

        Parameters:
        ----------
        data: numpy.ndarray
            dtype is float_ | uint8 | int32 | int_.
        """
        pass

    def scale_from(self, data: numpy.ndarray[Any, numpy.dtype[numpy.any]], scale: float) -> None:
        """ 
        Scale data to tensor in system memory

        Parameters:
        ----------
        data : ndarray
            Data of type float32 to be scaled from. 
        scale : float
            Scale value. 
        """
        pass

    def scale_to(self, scale: float, shape: list[int] = None) -> numpy.ndarray[Any, numpy.dtype[Any]] :
        """ 
        Scale tensor to data in system memory.

        Parameters:
        ----------
        scale : float
            Scale value. 
        shape : list[int], optional
            If provided, Shape of output data to scale to. 

        """
        pass

    def sync_s2d(self, size: int = None) -> None:
        """ 
        move size data from system to device

        Parameters:
        ----------
        size : int, optional
            If provided, byte size to be copied.
            else, move all data from system to device.
        """
        pass

    def sync_d2s(self, size: int = None) : 
        """ 
        move size data from device to system

        Parameters:
        ----------
        size : int, optional
            If provided, byte size to be copied.
            else, move size data from device to system
        """
        pass
    
    def pysys_data(self) ->  numpy.ndarray[Any, numpy.dtype[numpy.int32]] :   pass

    def memory_set(self, c: int) -> None : 
        """
        Fill memory with a comstant byte

        Parameters:
        ----------
        c: int
            fills memory with the constant byte c
        """
        pass

class Engine:
    def __init__(self, dev_id: int) : pass
    def __init__(self, handle: Handle) : pass
    def __init__(self, bmodel_path: str, dev_id: int, mode:IOMode) : pass
    def __init__(self, bmodel_bytes: bytes, bmodel_size: int, dev_id: int, mode:IOMode) : pass

    def load(self, bmodel_path: str) -> bool: 
        """ Load bmodel from file """
        pass

    def load(self, bmodel_bytes: bytes, bmodel_size: int) -> bool: 
        """ load bmodel from system memory """
        pass

    def get_handle(self) -> Handle:
        """Get Handle instance """
        pass

    def get_device_id(self) -> int:
        """Get device id of this engine """
        pass

    def get_graph_names(self) -> list[str]:
        """Get all graph names in the loaded bomodels """
        pass

    def set_io_mode(self, graph_name: str, mode: IOMode) -> bool:
        """Set IOMode for a graph """
        pass

    def get_input_names(self, graph_name: str) -> list[str]:
        """Get all input tensor names of the specified graph """
        pass

    def get_output_names(self, graph_name: str) -> list[str]:
        """Get all output tensor names of the specified graph """
        pass
    
    def get_max_input_shapes(self, graph_name: str) -> dict[str, list[int]]:
        """Get max shapes of input tensors in a graph """
        pass

    def get_input_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an input tensor in a graph """
        pass

    def get_max_output_shapes(self, graph_name: str) -> dict[str, list[int]]:
        """Get max shapes of output tensors in a graph """
        pass
    
    def get_output_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an output tensor in a graph """
        pass
    
    def get_input_dtype(self, graph_name: str, tensor_name: str) -> Dtype:
        """Get data type of an input tensor"""
        pass
    
    def get_output_dtype(self, graph_name: str, tensor_name: str) -> Dtype:
        """Get data type of an output tensor"""
        pass

    def get_input_scale(self, graph_name: str, tensor_name: str) -> float:
        """Get scale of an input tensor"""
        pass

    def get_output_scale(self, graph_name: str, tensor_name: str) -> float:
        """Get scale of an output tensor"""
        pass

    def process(self, graph_name: str, 
        input: dict[str, Tensor], 
        output: dict[str, Tensor]) -> None:
        """
        Inference with provided input and output tensors

        Parameters:
        ----------
        graph_name: str
            The specified graph name.
        input : dict[str, Tensor]
            Input tensors.
        output : dict[str, Tensor]
            Output tensors.
        """
        pass
    
    def process(self, graph_name: str, 
        input: dict[str, Tensor], 
        input_shapes: dict[str, list[int]],
        output: dict[str, Tensor]) -> None:
        """
        Inference with provided input and output tensors and input shapes.
        
        Parameters:
        ----------
        graph_name: str
            The specified graph name.
        input : dict[str, Tensor]
            Input tensors.
        input_shapes : dict[str, list[int]]  
            Real input tensor shapes.
        output : dict[str, Tensor]
            Output tensors.
        """
        pass
    def process(self, graph_name: str,
        input_tensors: dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]]) -> dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]] :
        """
        Inference with provided input.

        Parameters:
        ----------
        graph_name : str
            The specified graph name.
        input_tensors: dict[str,ndarray]
            Input tensors.

        Returns
        -------
        dict[str,ndarray]
        """
        pass
    
    def create_input_tensors_map(self, graph_name: str, create_mode: int = -1) -> dict[str,Tensor]:
        """
        Create input tensors map, according to and bmodel.

        Parameters:
        ----------
        graph_name : str
            The specified graph name.
        create_mode: Tensor Create mode
            case 0: only allocate system memory 
            case 1: only allocate device memory
            case other: according to engine IOMode

        Returns
        -------
        dict[str,Tensor]
        """
        pass

    def create_output_tensors_map(self, graph_name: str, create_mode: int = -1) -> dict[str,Tensor]:
        """
        Create output tensors map, according to and bmodel.

        Parameters:
        ----------
        graph_name : str
            The specified graph name.
        create_mode: Tensor Create mode
            case 0: only allocate system memory 
            case 1: only allocate device memory
            case other: according to engine IOMode

        Returns
        -------
        dict[str,Tensor]
        """
        pass  

class PaddingAtrr:
    def __init__(self): pass
    def __init__(self,stx: int, sty: int, w: int, h: int, \
        r: int, g: int, b: int): pass
    def set_stx(self, stx: int) -> None : pass
    def set_sty(self, sty: int) -> None : pass
    def set_w(self, w: int) -> None : pass
    def set_h(self, h: int) -> None : pass
    def set_r(self, r: int) -> None : pass
    def set_g(self, g: int) -> None : pass
    def set_b(self, b: int) -> None : pass


class Bmcv:
    def __init__(self, handle: Handle) -> None: pass

    def bm_image_to_tensor(self, img: BMImage|BMImageArray, tensor: Tensor) -> None:
        """
        Convert BMImage|BMImageArray to tensor

        Parameters:
        ----------
        img : BMImage|BMImageArray(Input image)

        Returns
        -------
        tensor: Tensor(Output tensor)
        """
        pass

    def bm_image_to_tensor(self, img: BMImage|BMImageArray) -> Tensor:
        """
        Convert BMImage|BMImageArray to tensor

        Parameters:
        ----------
        img : BMImage|BMImageArray(Input image)

        Returns
        -------
        tensor: Tensor(Output tensor)
        """
        pass

    def tensor_to_bm_image(self, tensor: Tensor, img: BMImage|BMImageArray, bgr2rgb: bool = False) -> None:
        """
        Convert tensor to BMImage|BMImageArray

        Parameters:
        ----------
        tensor : Tensor
            Input tensor.
        bgr2rgb : bool, optional
            Swap color channel, default is False.

        Returns
        -------
        img: BMImage|BMImageArray
            Output image
        """
        pass

    def tensor_to_bm_image(self, tensor: Tensor, bgr2rgb: bool = False) -> BMImage:
        """
        Convert tensor to BMImage

        Parameters:
        ----------
        tensor : Tensor
            Input tensor.
        bgr2rgb : bool, optional
            Swap color channel, default is False.

        Returns
        -------
        img: BMImage
            Output image
        """
        pass

    def crop_and_resize(self, input: BMImage|BMImageArray,
        crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, resize_w: int, resize_h: int, 
        resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Crop then resize an image. """
        pass
    
    def crop(self, input: BMImage|BMImageArray, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> BMImage|BMImageArray:
        """Crop an image with given window. """
        pass

    def resize(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Resize an image with interpolation of INTER_NEAREST. """
        pass

    def vpp_crop_and_resize(self, input: BMImage|BMImageArray,
        crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, resize_w: int, resize_h: int) -> BMImage|BMImageArray:
        """Crop then resize an image using vpp. """
        pass

    def vpp_crop(self, input: BMImage|BMImageArray, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int) -> BMImage|BMImageArray:
        """Crop an image with given window using vpp. """
        pass

    def vpp_resize(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, resize_w: int, resize_h: int) -> None:
        """Resize an image with interpolation of INTER_NEAREST using vpp. """
        pass
    
    def vpp_resize(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int) -> BMImage|BMImageArray:
        """Resize an image with interpolation of INTER_NEAREST using vpp. """
        pass

    def vpp_crop_and_resize_padding(self, input: BMImage|BMImageArray, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, 
        resize_w: int, resize_h: int, padding_in: PaddingAtrr, resize_alg: bmcv_resize_algorithm = BMCV_INTER_NEAREST) -> BMImage|BMImageArray:
        """Crop then resize an image using vpp. """
        pass
    
    def crop_and_resize_padding(self, input: BMImage, crop_x0: int, crop_y0: int, crop_w: int, crop_h: int, 
        resize_w: int, resize_h: int, padding_in: PaddingAtrr) -> BMImage:
        """Crop then resize an image. """
        pass

    def vpp_crop_padding(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, padding_in: PaddingAtrr) -> BMImage|BMImageArray:
        """Crop an image with given window using vpp. """
        pass

    def vpp_resize_padding(self, input: BMImage|BMImageArray, resize_w: int, resize_h: int, padding_in: PaddingAtrr) -> BMImage|BMImageArray:
        """Resize an image with interpolation of INTER_NEAREST using vpp. """
        pass

    def yuv2bgr(self, input: BMImage|BMImageArray) -> BMImage|BMImageArray: 
        """Convert an image from YUV to BGR."""
        pass

    def warp(self, input: BMImage|BMImageArray, matrix) -> BMImage|BMImageArray:
        """
        Applies an affine transformation to an image.
        
        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        matrix: 2x3 transformation matrix

        Returns:
        -------
        Output image
        """
        pass
    
    def warp_perspective(self, input: BMImage, coordinate, output_width: int,  output_height: int, \
        format: Format = Format.FORMAT_BGR_PLANAR,  dtype: ImgDtype = ImgDtype.DATA_TYPE_EXT_1N_BYTE, \
        use_bilinear: int = 0 ) -> BMImage:
        """
        Applies a perspective transformation to an image.
        
        Parameters:
        ----------
        input: BMImage
        coordinate: Original coordinate, like(left_top.x, left_top.y), (right_top.x, right_top.y), \
            (left_bottom.x, left_bottom.y), (right_bottom.x, right_bottom.y)
        output_width: Output width
        output_height: Output height
        Format: Output image format, Only FORMAT_BGR_PLANAR,FORMAT_RGB_PLANAR 
        dtype: Output image dtype, Only DATA_TYPE_EXT_1N_BYTE,DATA_TYPE_EXT_4N_BYTE
        use_bilinear: Bilinear use flag.

        Returns:
        -------
        Output image
        """
        pass

    def convert_to(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, alpha_beta) -> None:
        """
        Applies a linear transformation to an image.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

        Returns:
        output: BMImage|BMImageArray(Output image)
        """
        pass

    def convert_to(self, input: BMImage|BMImageArray, alpha_beta) -> BMImage|BMImageArray:
        """
        Applies a linear transformation to an image.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        alpha_beta:  (a0, b0), (a1, b1), (a2, b2) factors

        Returns:
        output: BMImage|BMImageArray(Output image)
        """
        pass

    def rectangle(self, image: BMImage, x: int, y: int, w: int, h: int, color, thickness: int = 1) -> None:
        """
        Draw a rectangle on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        x: int, Start point x of rectangle
        y: int, Start point y of rectangle
        w: int, Width of rectangle
        h: int, Height of rectangle
        color: Color of rectangle, like (0, 0, 255)
        thickness: int, optional, default is 1
        """
        pass

    def rectangle_(self, image: bm_image, x: int, y: int, w: int, h: int, color, thickness: int = 1) -> None:
        """
        Draw a rectangle on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        x: int, Start point x of rectangle
        y: int, Start point y of rectangle
        w: int, Width of rectangle
        h: int, Height of rectangle
        color: Color of rectangle, like (0, 0, 255)
        thickness: int, optional, default is 1
        """
        pass

    def putText(self, image: BMImage, text: str, x: int, y: int, color, fontScale: float, thickness: int=1) -> None:
        """
        Draw Text on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        text: str, Text string to be drawn
        x: int, Start point x
        y: int, Start point y
        color: color of text, like(0, 0, 255)
        fontScale: float, Font scale factor that is multiplied by the font-specific base size
        thickness: int, optional, default is 1
        """
        pass

    def putText_(self, image: bm_image, text: str, x: int, y: int, color, fontScale: float, thickness: int=1) -> None:
        """
        Draw Text on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        text: str, Text string to be drawn
        x: int, Start point x
        y: int, Start point y
        color: color of text, like(0, 0, 255)
        fontScale: float, Font scale factor that is multiplied by the font-specific base size
        thickness: int, optional, default is 1
        """
        pass


    def imwrite(self, filename: str, image: BMImage) -> None:
        """
        Save the image to the specified file.

        Parameters:
        ----------
        filename: str
            Name of the save file.
        image: BMImage
            Image to be saved.
        """
        pass

    def imwrite_(self, filename: str, image: bm_image) -> None:
        """
        Save the image to the specified file.
        
        Parameters:
        ----------
        filename: str
            Name of the save file.
        image: bm_image
            Image to be saved.
        """
        pass

    def get_handle(self) -> Handle: 
        """Get Handle instance. """
        pass

    def convert_format(self, input: BMImage, output: BMImage) -> None: 
        """Convert input to output format. """
        pass

    def convert_format(self, input: BMImage) -> BMImage: 
        """Convert an image to BGR PLANAR format. """
        pass

    def vpp_convert_format(self, input: BMImage, output: BMImage) -> None: 
        """Convert input to output format using vpp. """
        pass

    def vpp_convert_format(self, input: BMImage) -> BMImage: 
        """Convert an image to BGR PLANAR format using vpp. """
        pass

    def get_bm_data_type(self, format: ImgDtype) -> Dtype: pass

    def get_bm_image_data_format(self, dtype: Dtype) -> ImgDtype: pass

    def image_add_weighted(self, input1: BMImage, alpha: float, input2: BMImage, \
        beta: float, gamma: float, output: BMImage) -> int:  
        """output = input1 * alpha + input2 * beta + gamma."""
        pass

    def image_add_weighted(self, input1: BMImage, alpha: float, input2: BMImage, \
        beta: float, gamma: float) -> BMImage:
        """output = input1 * alpha + input2 * beta + gamma."""
        pass

    def image_copy_to(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, \
        start_x: int, start_y: int) -> None:
        """
        Copy the input to the output.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        output: BMImage|BMImageArray(Output image)
        start_x: point start x
        start_y: point start y
        """
        pass

    def image_copy_to_padding(self, input: BMImage|BMImageArray, output: BMImage|BMImageArray, \
        padding_r: numpy.uint8, padding_g: numpy.uint8, padding_b: numpy.uint8, start_x: int, start_y: int) -> None:
        """
        Copy the input to the output width padding.

        Parameters:
        ----------
        input: BMImage|BMImageArray(Input image)
        output: BMImage|BMImageArray(Output image)
        padding_r: r value for padding
        padding_g: g value for padding
        padding_b: b value for padding
        start_x: point start x
        start_y: point start y
        """
        pass

    def nms(self, input: numpy.ndarray[Any, numpy.dtype[numpy.float32]], threshold: float) -> numpy.ndarray[Any, numpy.dtype[numpy.float32]]:
        """
        Do nms use tpu.

        Parameters:
        ----------
        input: input proposal array
            shape must be (n,5) n<56000, proposal:[left,top,right,bottom,score]
        threshold: nms threshold

        Returns:
        ----------
        return nms result, numpy.ndarray[Any, numpy.dtype[numpy.float32]]
        """
        pass

    def drawPoint(self, image: BMImage, center: Tuple[int, int], color: Tuple[int, int, int], radius: int) -> int:
        """
        Draw Point on input image.

        Parameters:
        ----------
        image: BMImage, Input image
        center: Tuple[int, int], center of point, (point_x, point_y)
        color: Tuple[int, int, int], color of drawn, (b,g,r)
        radius: Radius of drawn
        """
        pass

    def drawPoint_(self, image: bm_image, center: Tuple[int, int], color: Tuple[int, int, int], radius: int) -> int:
        """
        Draw Point on input image.

        Parameters:
        ----------
        image: bm_image, Input image
        center: Tuple[int, int], center of point, (point_x, point_y)
        color: Tuple[int, int, int], color of drawn, (b,g,r)
        radius: Radius of drawn
        """
        pass

    def imdecode(self, data_bytes: bytes) -> BMImage:
        """
        Load Jpeg image from system memory.

        Parameters:
        ----------
        data_bytes: image data bytes in system memory


        Returns:
        ----------
        return decoded image
        """
        pass
    
    def fft(self, forward: bool, input_real: Tensor) -> list[Tensor]:
        """
        1d or 2d fft (only real part)
        
        Parameters:
        ----------
        forward: bool, positive transfer
        input_real: Tensor, input tensor
        
        
        Returns:
        ----------
        return list[Tensor], The real and imaginary part of output
        """
        pass
    
    def fft(self, forward: bool, input_real: Tensor, input_imag: Tensor) -> list[Tensor]:
        """
        1d or 2d fft (real part and imaginary part)
        
        Parameters:
        ----------
        forward: bool, positive transfer
        input_real: Tensor, input tensor real part
        input_imag: Tensor, input tensor imaginary part

        
        Returns:
        ----------
        return list[Tensor], The real and imaginary part of output
        """
        pass

class BMImageArray():
    def __init__(self) -> None: ...

    def __init__(self, handle: Handle, h: int, w: int, format: Format, dtype: ImgDtype ) -> None: ...

    def __len__(self) -> int: pass

    def __getitem__(self, i: int) -> bm_image: pass

    def __setitem__(self, i: int, data: bm_image) -> None: 
        """
        Copy the image to the specified index.

        Parameters:
        ----------
        i: int
            Index of the specified location.
        data: bm_image
            Input image.
        """
        pass

    def copy_from(self, i: int, data:BMImage) -> None: 
        """
        Copy the image to the specified index.

        Parameters:
        ----------
        i: int
            Index of the specified location.
        data: bm_image
            Input image.
        """
        pass

    def attach_from(self, i: int, data: BMImage) -> None: 
        """
        Attach the image to the specified index.(Because there is no memory copy, the original data needs to be cached)

        Parameters:
        ----------
        i: int
            Index of the specified location.
        data: BMImage
            Input image.
        """
        pass

    def get_device_id(self) -> int: pass


class BMImageArray2D(BMImageArray): pass
class BMImageArray3D(BMImageArray): pass
class BMImageArray4D(BMImageArray): pass
class BMImageArray8D(BMImageArray): pass
class BMImageArray16D(BMImageArray): pass
class BMImageArray32D(BMImageArray): pass
class BMImageArray64D(BMImageArray): pass
class BMImageArray128D(BMImageArray): pass
class BMImageArray256D(BMImageArray): pass
    
class MultiEngine:
    def __init__(self, bmodel_path: str, dev_ids: list[int], sys_out: bool = True, graph_idx: int = 0) : pass

    def set_print_flag(self, flag:bool) -> None :pass

    def set_print_time(self, flag:bool) -> None :pass
    
    def get_device_ids(self) -> list[int]:
        """Get device ids of this MultiEngine """
        pass

    def get_graph_names(self) -> list[str]:
        """Get all graph names in the loaded bomodels """
        pass  

    def get_input_names(self, graph_name: str) -> list[str]:
        """Get all input tensor names of the specified graph """
        pass

    def get_output_names(self, graph_name: str) -> list[str]:
        """Get all output tensor names of the specified graph """
        pass

    def get_input_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an input tensor in a graph """
        pass

    def get_output_shape(self, graph_name: str, tensor_name: str) -> list[int]:
        """Get the shape of an output tensor in a graph """
        pass

    def process(self,
        input_tensors: dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]]) -> dict[str, numpy.ndarray[Any, numpy.dtype[numpy.float_]]] :
        """
        Inference with provided input.

        Parameters:
        ----------
        input_tensors: dict[str,ndarray]
            Input tensors.

        Returns
        -------
        dict[str,ndarray]
        """
        pass    

class MultiDecoder:

    def __init__(self, queue_size: int = 10, tpu_id: int = 0, discard_mode: int = 0): 
        """ MultiDecoder Constructor.

        Parameters:
        ----------
        queue_size: int
            Max queue size,default is 10.
        tpu_id: int
            ID of TPU, default is 0.
        discard_mode: int
            Data discard policy when the queue is full. Default is 0.
            If 0, do not push the data to queue, else pop the data from queue and push new data to queue.
        """
        pass


    def set_read_timeout(self, timeout: int) -> None : 

        """ Set read frame timeout waiting time 
        
        Parameters:
        ----------
        timeout: int
            Set read frame timeout waiting time in seconds.
        """
        pass


    def add_channel(self, file_path: str, frame_skip_num: int = 0) -> int:
        
        """ Add a channel to decode

        Parameters:
        ----------
        file_path : str
            file path
        frame_skip_num : int
            frame skip number, default is 0

        Returns:
        ----------
        return channel index number, int
        """
        pass

    def del_channel(self, channel_idx: int) -> int : 
        """ Delete channel
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        0 for success and other for failure.
        """
        pass

    def clear_queue(self, channel_idx: int) -> int : 
        """ Clear data cache queue 
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        0 for success and other for failure.
        """
        pass

    def read(self, channel_idx: int, image: BMImage, read_mode: int = 0) -> int : 
        """ Read a BMImage from the MultiDecoder with a given channel.

        Parameters:
        ----------
        channel_idx : int
            channel index number
        image: BMImage
            BMImage instance to be read to
        read_mode: int
            Read data mode, 0 for not waiting data and other waiting data.

        Returns:
        -------
        0 for successed get data.
        """
        pass

    def read(self,channel_idx: int) -> int :
        """ Read a BMImage from the MultiDecoder with a given channel.

        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
            BMImage instance to be read to
        """
        pass


    def read_(self, channel_idx: int, image: bm_image, read_mode: int=0) -> int :
        """ Read a bm_image from the MultiDecoder with a given channel.

        Parameters:
        ----------
        channel_idx : int
            channel index number
        image: bm_image
            bm_image instance to be read to
        read_mode: int
            read data mode, default 0, if 0, not waiting data, else waiting data.

        Returns:
        -------
        return 0 if get data.
        """
        pass

    def read_(self,channel_idx: int) -> int :
        """ Read a bm_image from the MultiDecoder with a given channel.

        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
            bm_image instance to be read to
        """
        pass

        
    def reconnect(self, channel_idx: int) -> int :
        """ Reconnect Decoder for instance channel
            
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns:
        -------
        0 for success and other for failure.
            
        """
        pass


    def get_frame_shape(self, channel_idx: int) -> list[int]:
        """ Get frame for instance channel
        
        Parameters:
        ----------
        channel_idx : int
            channel index number

        Returns
        -------
        list[int], [1, C, H, W]
        """
        pass

    def set_local_flag(self, flag: bool) -> None: 
        """ Set local video flag

        Parameters:
        ----------
        flag : bool
            If flag is True, Decode up to 25 frames per second

        """
        pass

class sail_resize_type(enum.Enum):
    BM_RESIZE_VPP_NEAREST = 0
    BM_RESIZE_TPU_NEAREST = 1
    BM_RESIZE_TPU_LINEAR = 2
    BM_RESIZE_TPU_BICUBIC = 3
    BM_PADDING_VPP_NEAREST = 4
    BM_PADDING_TPU_NEAREST = 5
    BM_PADDING_TPU_LINEAR = 6
    BM_PADDING_TPU_BICUBIC = 7

class ImagePreProcess:
    
    def __init__(self, batch_size: int, resize_mode:sail_resize_type, tpu_id: int=0, 
        queue_in_size: int=20, queue_out_size: int=20):  
        """ ImagePreProcess Constructor.

        Parameters:
        ----------
        batch_size: int
            Output batch size.
        resize_mode: sail_resize_type
            Resize Methods
        tpu_id: int
            ID of TPU, there may be more than one TPU for PCIE mode,default is 0.
        queue_in_size: int
            Max input image data queue size, default is 20.
        queue_out_size: int
            Max output tensor data queue size, default is 20.
        """
        pass


    def SetResizeImageAtrr(self, output_width: int, output_height: int ,bgr2rgb : bool, dtype: ImgDtype) -> None : 

        """ Set the Resize Image attribute 
        
        Parameters:
        ----------
        output_width: int
            The width of resized image.
        output_height: int
            The height of resized image.
        bgr2rgb: bool
            The flag of convert BGR image to RGB.
        dtype: ImgDtype  
            The data type of resized image,Only supported BM_FLOAT32,BM_INT8,BM_UINT8   
        """
        pass


    def SetPaddingAtrr(self,padding_b:int=114,padding_g:int=114,padding_r:int=114,align:int=0) -> None :
        """ Set the padding attribute object.

        Parameters:
        ----------
        padding_b: int
            padding value of b channel, dafault 114
        padding_g: int
            padding value of g channel, dafault 114
        padding_r: int
            padding value of r channel, dafault 114
        align: int
            padding position, default 0: start left top, 1 for center

        """
        pass


    def SetConvertAtrr(self,alpha_beta) -> int :
        """ Set the linear transformation attribute 

        Parameters:
        ----------
        alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

        Returns
        -------
        0 for success and other for failure 
        """
        pass


    def PushImage(self, channel_idx : int, image_idx : int, image: BMImage) -> int: 
        """ Push Image
        Parameters:
        ----------
        channel_idx : int
            Channel index number of the image
        image_idx: int
            Image index number of the image
        image: BMImage
            Input image
        
        Returns
        -------
        0 for success and other for failure 
        """
        pass

    def GetBatchData(self) -> tuple[Tensor, list[BMImage],list[int],list[int],list[list[int]]]:
        """ Get the Batch Data object
        
        Returns
        -------
        [Tensor, list[BMImage],list[int],list[int],list[list[int]]]
            Output Tensor, Original Images, Original Channel index, Original Index, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def set_print_flag(self, flag : bool) -> None:
        """ Set the print flag
        
        Parameters:
        ----------
        flag : int
        """
        pass

class TensorPTRWithName:
    def get_name(self) -> str:
        """ Get the name of the Tensor.

        pass
        """

    def get_data(self) -> Tensor:
        """ Get the Tensor.
        
        pass
        """
    

class EngineImagePreProcess:
    
    def __init__(self, bmodel_path: str, tpu_id: int, use_mat_output: bool = False): 
        """ EngineImagePreProcess Constructor.

        Parameters:
        ----------
        bmodel_path: str
        tpu_id: int
            ID of TPU, there may be more than one TPU for PCIE mode,default is 0.
        use_mat_output : bool
            Use opencv Mat for output images
        """
        pass

    def InitImagePreProcess(self, resize_mode: sail_resize_type, bgr2rgb: bool = False,
        queue_in_size: int = 20, queue_out_size: int = 20) -> int:
        """ initialize ImagePreProcess.

        Parameters:
        ----------
        resize_mode: sail_resize_type
            Resize Methods
        bgr2rgb: bool
            The flag of convert BGR image to RGB, default is False.
        queue_in_size: int
            Max input image data queue size, default is 20.
        queue_out_size: int
            Max output tensor data queue size, default is 20.
        
        Returns
        -------
        0 for success and other for failure 
        """
        pass


    def SetPaddingAtrr(self,padding_b:int=114,padding_g:int=114,padding_r:int=114,align:int=0) -> int :
        """ Set the padding attribute object.

        Parameters:
        ----------
        padding_b: int
            padding value of b channel, dafault 114
        padding_g: int
            padding value of g channel, dafault 114
        padding_r: int
            padding value of r channel, dafault 114
        align: int
            padding position, default 0: start left top, 1 for center
        
        Returns
        -------
        0 for success and other for failure 
        """

    def SetConvertAtrr(self,alpha_beta) -> int :
        """ Set the linear transformation attribute 

        Parameters:
        ----------
        alpha_beta:like (a0, b0), (a1, b1), (a2, b2) factors

        Returns
        -------
        0 for success and other for failure 
        """
        pass

    def PushImage(self, channel_idx : int, image_idx : int, image: BMImage) -> int: 
        """ Push Image
        Parameters:
        ----------
        channel_idx : int
            Channel index number of the image
        image_idx: int
            Image index number of the image
        image: BMImage
            Input image
        
        Returns
        -------
        0 for success and other for failure 
        """
        pass


    def GetBatchData_Npy(self) -> tuple:
        """ Get the Batch Data object
        
        Returns
        -------
        [dict[str, ndarray], list[BMImage],list[int],list[int],list[list[int]]]
            Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def GetBatchData_Npy2(self) -> tuple:
        """ Get the Batch Data object
        
        Returns
        -------
        [dict[str, ndarray], list[numpy.ndarray[numpy.uint8]],list[int],list[int],list[list[int]]]
            Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def GetBatchData(self, need_d2s: bool = True) -> tuple:
        """ Get the Batch Data object
        
        Parameters:
        ------------
        need_d2s : bool
            Need copy data to system memory.

        Returns
        -------
        [list[TensorPTRWithName], list[BMImage],list[int],list[int],list[list[int]]]
            Output ndarray map, Original Images, Original Channel index, Original Indexm, Padding Atrr(start_x, start_y, width, height)
        """
        pass

    def get_graph_name(self) -> str:
        """ Get first graph name in the loaded bomodel
        
        Returns
        -------
        First graph name
        """
        pass

    def get_input_width(self) -> int:
        """ Get model input width

        Returns
        -------
        Model input width
        """
        pass

    def get_input_height(self) -> int:
        """ Get model input height

        Returns
        -------
        Model input height
        """
        pass

    def get_output_names(self) -> list[str]:
        """ Get all output tensor names of the first graph

        Returns
        -------
        All the output tensor names of the graph
        """
        pass
    
    def get_output_shape(self, tensor_name: str) -> list[int]:
        """ Get the shape of an output tensor in frist graph
        
        Parameters:
        ----------
        tensor_name : str
             The specified tensor name

        Returns
        -------
        The shape of the tensor
        """
        pass

class algo_yolov5_post_1output:
    
    def __init__(self, shape: list[int], network_w:int = 640, network_h:int = 640, max_queue_size: int=20): 
        """ algo_yolov5_post_1output Constructor.

        Parameters:
        ----------
        shape: list[int]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        """
        pass

    def push_npy(self, 
                channel_idx : int, 
                image_idx : int, 
                data : numpy.ndarray[Any, numpy.dtype[numpy.float_]], 
                dete_threshold : float, 
                nms_threshold : float,
                ost_w : int, 
                ost_h : int,
                padding_left : int,
                padding_top : int,
                padding_width : int,
                padding_height : int) -> int:
        """ algo_yolov5_post_1output push data.

        Parameters:
        ----------
        
        channel_idx : Channel index number of the image.
        image_idx : Image index number of the image.
        data : Input Data
        dete_threshold : Detection threshold
        nms_threshold : NMS threshold
        ost_w : Original image width
        ost_h : Original image height
        padding_left : Padding left
        padding_top : Padding top
        padding_width : Padding width
        padding_height : Padding height

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : TensorPTRWithName, 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov5_post_1output push data.

        Parameters:
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class algo_yolov5_post_3output:
    
    def __init__(self, shape: list[list[int]], network_w:int = 640, network_h:int = 640, max_queue_size: int=20): 
        """ algo_yolov5_post_1output Constructor.

        Parameters:
        ----------
        shape: list[list[int]]
        network_w: int, network input width 
        network_h: int, network input height 
        max_queue_size: int, default 20
        """
        pass
    
    def reset_anchors(self, anchors_new: list[list[list[int]]]):
        """ Reset Anchors
        
        Parameters:
        ----------
        anchors_new: list[list[list[int]]]
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass
    
    def get_result_npy(self) -> tuple[tuple[int, int, int, int, int, float],int, int]:
        """ Get the PostProcess result
        
        Returns
        -------
        tuple[tuple[left, top, right, bottom, class_id, score],channel_idx, image_idx]
            
        """
        pass


    def push_data( self, 
        channel_idx : list[int], 
        image_idx : list[int], 
        input_data : list[TensorPTRWithName], 
        dete_threshold : list[float],
        nms_threshold : list[float],
        ost_w : list[int],
        ost_h : list[int],
        padding_attr : list[list[int]]) -> int:
        """ algo_yolov5_post_1output push data.

        Parameters:
        ----------
        
        channel_idx : Channel index number of the images.
        image_idx : Image index number of the images.
        data : Input Data
        dete_threshold : Detection thresholds
        nms_threshold : NMS thresholds
        ost_w : Original images width
        ost_h : Original images height
        padding_attr : Padding Attribute[start_x, start_y, width, height]

        Returns
        -------
        return 0 for success and other for failure
        """
        pass

class Encoder:
    
    def __init__(self, handle: Handle) -> None:
        """
        encoder constructor for pic
        """
        pass
    
    def __init__(self, output_path: str, handle: Handle, enc_fmt: str, pix_fmt: str, enc_params: str) -> None:
        """
        encoder constructor for video
        
        Parameters:
        ----------
        output_path: local file path or stream url
        handle: sail.Handle
        enc_fmt: encoder name, support h264_bm and h265_bm
        pix_fmt: encoder pixel format, support I420 and NV12
        enc_params: encoder params, width=1920:height=1080:bitrate=2000:gop=32:gop_preset=2:framerate=25
        """
        pass
        
    def is_opened(self) -> bool:

        pass
    
    def pic_encode(self, ext: str, image: BMImage) -> numpy.ndarray[Any, numpy.dtype[Any]]:
        """
        pic encode
        
        Parameters:
        ----------
        ext: encode format, .jpg .png ...
        image: input image
        
        Returns:
        ----------
        return pic data numpy.ndarray[Any, numpy.dtype[numpy.uint8_t]]
        """
        pass
    
    def pic_encode(self, ext: str, image: bm_image) -> numpy.ndarray[Any, numpy.dtype[Any]]:
        
        pass
    
    def video_write(self, image: BMImage) -> int:
        """
        encode video, convert input format to I420 or NV12. support rgb and yuv420, yuv422, yuv444.
        """
        pass
    
    def video_write(self, image: bm_image) -> int:
        
        pass  
    
    def release(self) -> None:
        """
        release resources 
        only invoked when video encode finished
        """
        pass


class tpu_kernel_api_yolov5_detect_out:
    
    def __init__(self, device_id: int, shapes: list[list[int]], network_w:int = 640, network_h:int = 640, 
                module_file: str='/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so'): 

        """ tpu_kernel_api_yolov5_detect_out Constructor.

        Parameters:
        ----------
        device_id: int, Device id
        shapes: list[list[int]],Input Data shape
        network_w: int, Network input width 
        network_h: int, Network input height 
        module_file: str, TPU Kernel module file,default is '/opt/sophon/libsophon-current/lib/tpu_module/libbm1684x_kernel_module.so'
        """
        pass


    def process(self,input:list[TensorPTRWithName], dete_threshold: float, nms_threshold: float) -> list[list[tuple[int, int, int, int, int, float]]]:
        """ Process

        Parameters:
        ----------
        input: list[TensorPTRWithName], Input Data 
        dete_threshold: float, Detection threshold
        nms_threshold: float, NMS threshold

        Returns
        -------
        list[list[tuple[left, top, right, bottom, class_id, score]]]
            
        """

    def process(self,input:dict[str, Tensor], dete_threshold: float, nms_threshold: float) -> list[list[tuple[int, int, int, int, int, float]]]:
        """ Process

        Parameters:
        ----------
        input: dict[str, Tensor], Input Data 
        dete_threshold: float, Detection threshold
        nms_threshold: float, NMS threshold

        Returns
        -------
        list[list[tuple[left, top, right, bottom, class_id, score]]]
            
        """
                
    def reset_anchors(self, anchors_new: list[list[list[int]]]):

        """ Reset Anchors
        
        Parameters:
        ----------
        anchors_new: list[list[list[int]]]
        
        Returns
        -------
        return 0 for success and other for failure
        """
        pass