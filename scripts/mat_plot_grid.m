function mat_plot_grid(phantom_x, phantom_y, phantom_z, ...
                       res_res, res_mat_fpath, res_img_fpath)
%% Plotter
load(res_mat_fpath);
% flr_pred_grid = reshape(flr_pred_grid, [res_res*phantom_x, res_res*phantom_y, res_res*phantom_z]);
figure;
phantom_z = double(phantom_z);
for z = 1:phantom_z+1
% for z = 1:phantom_z
    if phantom_z <= 8
        subplot(3,3,z); imagesc(flr_pred_grid(:,:,z), [0 max(flr_pred_grid(:))]); % colorbar;
    elseif phantom_z <= 15
        subplot(4,4,z); imagesc(flr_pred_grid(:,:,z), [0 double(max(flr_pred_grid(:)))]); axis square; colorbar;
    else
        subplot(6,4,z); imagesc(flr_pred_grid(:,:,z), [0 max(flr_pred_grid(:))]); axis square; colorbar;
    end
end
saveas(gcf, res_img_fpath);
close(gcf);

%%
clear;
end