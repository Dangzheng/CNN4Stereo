-- data handler for kitti 2012 & 2015.
--
-- Wenjie Luo
--

require 'image'
require 'cutorch'
require 'gnuplot'
require 'xlua'

local DataHandler = torch.class('DataHandler')

function DataHandler:__init( data_version, data_root, util_root, num_tr_img, num_val_img, num_val_loc, batch_size, psz, half_range, gpu)
    -- data_root: training image folder
    -- util_root: loc binary file, perm fn
    if data_version == 'kitti2015' then
        self.nChannel = 3
    elseif data_version == 'kitti2012' then
        self.nChannel = 1
    else
        error('data_version should be either kitti2012 or kitti2015')
    end

    self.batch_size = batch_size
    self.psz = psz
    self.pSize = 2*psz + 1
    self.half_range = half_range
    self.cuda = gpu or 1 -- use gpu in default
    self.tr_ptr = 0
    self.curr_epoch = 0

    self.file_ids = torch.FloatTensor(torch.FloatStorage(paths.concat(util_root, 'myPerm.bin')))
    self.tr_loc = torch.FloatTensor(torch.FloatStorage(('%s/tr_%d_%d_%d.bin'):format(util_root, num_tr_img, self.psz, self.half_range))):view(-1,5)
    if num_val_img == 0 then
        self.val_loc = self.tr_loc
        print('validate on training set..')
    else
        self.val_loc = torch.FloatTensor(torch.FloatStorage(('%s/val_%d_%d_%d.bin'):format(util_root, num_val_img, self.psz, self.half_range))):view(-1,5)
    end

    print(string.format('#training locations: %d -- #valuation locations: %d', (#self.tr_loc)[1], (#self.val_loc)[1]))

    self.ldata = {}
    self.rdata = {}

    --[[--]]
    self.tr_perm = torch.randperm((#self.tr_loc)[1])
    self.val_perm = torch.randperm((#self.val_loc)[1])
    
    for i=1, num_tr_img+num_val_img do
        xlua.progress(i, num_tr_img+num_val_img)
        local fn = self.file_ids[i]
        local l_img, r_img
        if data_version == 'kitti2015' then
            l_img = image.load(string.format('%s/image_2/%06d_10.png', data_root, fn), self.nChannel, 'byte'):float()
            r_img = image.load(string.format('%s/image_3/%06d_10.png', data_root, fn), self.nChannel, 'byte'):float()
        elseif data_version == 'kitti2012' then
            l_img = image.load(string.format('%s/image_0/%06d_10.png', data_root, fn), self.nChannel, 'byte'):float()
            r_img = image.load(string.format('%s/image_1/%06d_10.png', data_root, fn), self.nChannel, 'byte'):float()
        end        

        -- pre-process images
        l_img:add(-l_img:mean()):div(l_img:std())
        r_img:add(-r_img:mean()):div(r_img:std())

        self.ldata[fn] = l_img
        self.rdata[fn] = r_img
    end


    -- function reserve_memory()
        -- reserve memory for batch data and evaluation data.
    print(string.format('receptive field size: %d; total range: %d', self.pSize, self.half_range*2+1))
    self.batch_left = torch.Tensor(self.batch_size, self.nChannel, self.pSize, self.pSize)
    self.batch_right = torch.Tensor(self.batch_size, self.nChannel, self.pSize, self.pSize+self.half_range*2)
    self.batch_label = torch.Tensor(self.batch_size, 1):fill(self.half_range+1)

    self.val_left = torch.Tensor(num_val_loc, self.nChannel, self.pSize, self.pSize)
    self.val_right = torch.Tensor(num_val_loc, self.nChannel, self.pSize, self.pSize+self.half_range*2)
    self.val_label = torch.Tensor(num_val_loc, 1):fill(self.half_range+1)

    for i = 1, num_val_loc do
        local img_id, loc_type, center_x, center_y, right_center_x = self.val_loc[self.val_perm[i]][1], self.val_loc[self.val_perm[i]][2], self.val_loc[self.val_perm[i]][3], self.val_loc[self.val_perm[i]][4], self.val_loc[self.val_perm[i]][5]
        local right_center_y = center_y
        self.val_left[i] = self.ldata[img_id][{{}, {center_y-self.psz, center_y+self.psz}, {center_x-self.psz, center_x+self.psz}}]
        if loc_type == 1 then -- horizontal
            self.val_right[i] = self.rdata[img_id][{{}, {right_center_y-self.psz, right_center_y+self.psz}, {right_center_x-self.psz-self.half_range, right_center_x+self.psz+self.half_range}}]
        elseif loc_type == 2 then -- vertical
            self.val_right[i] = self.rdata[img_id][{{}, {right_center_y-self.psz-self.half_range, right_center_y+self.psz+self.half_range}, {right_center_x-self.psz, right_center_x+self.psz}}]:transpose(2,3)
        end
    end

    print(string.format('validation created: num(%d)', num_val_loc))
    -- end

    collectgarbage()
end

-- training on left patch and whole range of right patch
function DataHandler:next_batch()
    -- torch.manualSeed(234)
    -- self.current_idx = 1

    -- self.tr_ptr = 0 -- over-fitting check   

    for idx = 1, self.batch_size do
        local i = self.tr_ptr + idx
        if i > torch.numel(self.tr_perm) then
            i = 1
            self.tr_ptr = -idx + 1
            self.curr_epoch = self.curr_epoch + 1
            print('....epoch id: ' .. self.curr_epoch .. ' done ......\n')
        end
        
        local img_id, loc_type, center_x, center_y, right_center_x = self.tr_loc[self.tr_perm[i]][1], self.tr_loc[self.tr_perm[i]][2], self.tr_loc[self.tr_perm[i]][3], self.tr_loc[self.tr_perm[i]][4], self.tr_loc[self.tr_perm[i]][5]
        local right_center_y = center_y
        
        self.batch_left[idx] = self.ldata[img_id][{{}, {center_y-self.psz, center_y+self.psz}, {center_x-self.psz, center_x+self.psz}}]
        if loc_type == 1 then -- horizontal
            self.batch_right[idx] = self.rdata[img_id][{{}, {right_center_y-self.psz, right_center_y+self.psz}, {right_center_x-self.psz-self.half_range, right_center_x+self.psz+self.half_range}}]
        elseif loc_type == 2 then -- vertical
            self.batch_right[idx] = self.rdata[img_id][{{}, {right_center_y-self.psz-self.half_range, right_center_y+self.psz+self.half_range}, {right_center_x-self.psz, right_center_x+self.psz}}]:transpose(2,3)
        end
        -- label is always half_range + 1
    end

    self.tr_ptr = self.tr_ptr + self.batch_size
    
    if self.cuda == 1 then
        return self.batch_left:cuda(), self.batch_right:cuda(), self.batch_label:cuda()
    else
        return self.batch_left, self.batch_right, self.batch_label
    end

end 

function DataHandler:get_eval_cuda()
    return self.val_left:cuda(), self.val_right:cuda(), self.val_label:cuda()
end

