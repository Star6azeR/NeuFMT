function mat_plot_mesh(phantom_x, phantom_y, phantom_z, te_maxvol, node_size, ...
                       res_mat_fpath, res_img_fpath)
%% Create Mesh
pta = [0, 0, 0];
ptz = [phantom_x, phantom_y, phantom_z];
[node, face, elem] = meshabox(pta, ptz, te_maxvol, node_size);

eltp = 3 * ones(length(elem), 1);
tMesh = toastMesh(node, elem, eltp);
tBasis = toastBasis(tMesh, ptz+1);

%% Plotter
load(res_mat_fpath);
flr_pred = tBasis.Map('M->B', flr_pred);
flr_pred = reshape(flr_pred, ptz+1);
figure;
for z = 1:phantom_z
    if phantom_z <= 9
        subplot(3,3,z); imagesc(flr_pred(:,:,z), [0 max(flr_pred(:))]);
    elseif phantom_z <= 16
        subplot(4,4,z); imagesc(flr_pred(:,:,z), [0 max(flr_pred(:))]); axis square;
    else
        subplot(4,5,z); imagesc(flr_pred(:,:,z), [0 max(flr_pred(:))]); axis square;
    end
end
saveas(gcf, res_img_fpath);
close(gcf);

%%
clear;
end