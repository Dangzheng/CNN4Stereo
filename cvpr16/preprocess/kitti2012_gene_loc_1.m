function [ty1, ty2] = kitti2012_gene_loc_1(tr_num, val_num, psz, half_range, saveDir, seed)
addpath('devkit_2012')

total_num = 194;

if strcmp(computer, 'MACI64')
    data_root = '';
else
    data_root = '/u/wenjie/dataset/kitti_2012/training';
end

if ~exist('saveDir', 'dir')
    mkdir(saveDir);
end

% half_range = 100; % range = 2*half_range + 1
% psz = 13; % pSize = 2*psz + 1
num_type = 2;
noc_occ = 'noc';
rng(seed);
fn_idx = randperm(total_num)-1;
% fn_idx = load(perm_fn);

%%
num_loc = zeros(1, tr_num);
num_pixel = zeros(1, tr_num);

for idx = 1:tr_num
    fn = sprintf('%s/disp_%s/%06d_10.png', data_root, noc_occ, fn_idx(idx));
    tmp = disp_read(fn);
    num_loc(idx) = sum(tmp(:)~=-1);
    num_pixel(idx) = numel(tmp);
end

all_loc = zeros(5, num_type*sum(num_loc));

ty1 = 0;
ty2 = 0;
valid_count = 0;
for idx = 1:tr_num
    idx
    fn = sprintf('%s/disp_%s/%06d_10.png', data_root, noc_occ, fn_idx(idx));
    tmp = disp_read(fn);
    [r, c] = find(tmp~=-1);
    [img_h, img_w] = size(tmp);
    
    assert(numel(r) == num_loc(idx));
    
    for loc = 1:length(r)
        l_center_x = c(loc);
        l_center_y = r(loc);
        r_center_x = round(c(loc) - tmp(r(loc), c(loc))); % it's '-' not '+'
        r_center_y = l_center_y;
        ll = l_center_x+psz <= img_w && l_center_x-psz > 0 && l_center_y+psz <= img_h && l_center_y-psz > 0;
        % horizontal
        rr_type1 = r_center_x-half_range-psz > 0 && r_center_x+half_range+psz <= img_w && r_center_y-psz > 0 && r_center_y+psz <= img_h;
        % vertical
        rr_type2 = r_center_y-half_range-psz > 0 && r_center_y+half_range+psz <= img_h && r_center_x-psz > 0 && r_center_x+psz <= img_w;
        
        if ll && rr_type1
            ty1 = ty1 + 1;
            valid_count = valid_count + 1;
            all_loc(:,valid_count) = [fn_idx(idx); 1; l_center_x; l_center_y; r_center_x];
        end
%         if ll && rr_type2
%             ty2 = ty2 + 1;
%             valid_count = valid_count + 1;
%             all_loc(:,valid_count) = [fn_idx(idx); 2; l_center_x; l_center_y; r_center_x];
%         end
        
    end
end
all_loc = all_loc(:, 1:valid_count);

loc_fn = sprintf('%s/tr_%d_%d_%d.bin', saveDir, tr_num, psz, half_range);

fw_id = fopen(loc_fn, 'wb');
fwrite(fw_id, all_loc, 'float');
fclose(fw_id);

fprintf('traing -- percentage: %f, num loc: %d, valid pair: %d\n', valid_count/(num_type*sum(num_loc)), sum(num_loc), valid_count);


%%
num_loc = zeros(1, val_num);
num_pixel = zeros(1, val_num);

for idx = 1:val_num
    fn = sprintf('%s/disp_%s/%06d_10.png', data_root, noc_occ, fn_idx(tr_num+idx));
    tmp = disp_read(fn);
    num_loc(idx) = sum(tmp(:)~=-1);
    num_pixel(idx) = numel(tmp);
end

all_loc = zeros(5, num_type*sum(num_loc));

valid_count = 0;
for idx = 1:val_num
    idx
    fn = sprintf('%s/disp_%s/%06d_10.png', data_root, noc_occ, fn_idx(tr_num+idx));
    tmp = disp_read(fn);
    [r, c] = find(tmp~=-1);
    [img_h, img_w] = size(tmp);
    
    assert(numel(r) == num_loc(idx));
    
    for loc = 1:length(r)
        l_center_x = c(loc);
        l_center_y = r(loc);
        r_center_x = round(c(loc) - tmp(r(loc), c(loc))); % it's '-' not '+'
        r_center_y = l_center_y;
        ll = l_center_x+psz <= img_w && l_center_x-psz > 0 && l_center_y+psz <= img_h && l_center_y-psz > 0;
        % horizontal
        rr_type1 = r_center_x-half_range-psz > 0 && r_center_x+half_range+psz <= img_w && r_center_y-psz > 0 && r_center_y+psz <= img_h;
        % vertical
        rr_type2 = r_center_y-half_range-psz > 0 && r_center_y+half_range+psz <= img_h && r_center_x-psz > 0 && r_center_x+psz <= img_w;
        
        if ll && rr_type1
            valid_count = valid_count + 1;
            all_loc(:,valid_count) = [fn_idx(tr_num+idx); 1; l_center_x; l_center_y; r_center_x];
        end
%         if ll && rr_type2
%             valid_count = valid_count + 1;
%             all_loc(:,valid_count) = [fn_idx(tr_num+idx); 2; l_center_x; l_center_y; r_center_x];
%         end
        
    end
end
all_loc = all_loc(:,1:valid_count);

loc_fn = sprintf('%s/val_%d_%d_%d.bin', saveDir, val_num, psz, half_range);

fw_id = fopen(loc_fn, 'wb');
fwrite(fw_id, all_loc, 'float');
fclose(fw_id);

fprintf('validation -- percentage: %f, num loc: %d, valid pair: %d\n', valid_count/(num_type*sum(num_loc)), sum(num_loc), valid_count);

%%
% write permutation in binary
fw_id = fopen(sprintf('%s/myPerm.bin', saveDir), 'wb');
fwrite(fw_id, fn_idx, 'float');
fclose(fw_id);

end
