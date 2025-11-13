function [Iout,elapsed] = retinex_fu(img,img_name,play,savepath) 


ts = tic;
input = img; 

Iout = underwater(input);
elapsed = toc(ts);

% disp('Elapsed time in seconds : ')
% disp(elapsed)

if play
figure();
    if (size(Iout,3) == 1)
        colormap(gray)
    end
    imshow(Iout); title('Dehazed Image')
    figure;imshow(trans_refined); colormap('jet'); title('Transmission')
end

sspath=[savepath,img_name,'_retinex_fu.jpg'];
imwrite(Iout,sspath);


end