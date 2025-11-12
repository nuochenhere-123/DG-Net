

if __name__ == '__main__':
# resize回 原大小 保存
      result_h0 = (final_enhanced_img * 255).astype(np.uint8)
      a=Image.fromarray(result_h0)
      b=a.resize(original_size)
      original_size_enhanced = np.array(b)
      io.imsave(image_path1, original_size_enhanced) 
      print(str(self.id)+": "+image_path1)