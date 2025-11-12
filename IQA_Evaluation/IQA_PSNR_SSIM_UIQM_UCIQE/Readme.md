### 有参考图像测试（PSNR & SSIM & UIQM & UCIQE）：  
python **PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py** {Testingset} {reference}  
e.g: python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\UWMamba-110-256 .\110_reference_256  
python PSNR_SSIM_UIQM_UCIQE_inLAB_ucolor.py .\RetinexBased-110 .\UIEB-reference-890   

### 无参考图像(UIQM & UCIQE)：  
python python **UCIQE_UIQM_inLAB.py** {Testingset}  
e.g: python python UCIQE_UIQM_inLAB.py .\upgrade2_3-U45  
python UCIQE_UIQM_inLAB.py ./RetinexBased-C60

