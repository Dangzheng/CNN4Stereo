require 'xlua'
require 'optim'
require 'cunn'
require 'image'
require 'gnuplot'

local c = require 'trepl.colorize'

opt = lapp[[
    --model                 (default 'split_win19_dep9')             model name
    -g, --gpuid             (default 0)                  gpu id
    --feat_dim              (default 64)

    --data_version          (default 'kitti2015')   
    --data_root             (default '/ais/gobi3/datasets/kitti/scene_flow')
    --perm_fn               (default '')

    --model_param           (default '')    weight file
    --bn_meanstd            (default '')

    --saveDir               (default 'debug')   folder for intermediate prediction result
    
    --sub_width             (default 2000)

    --start_id              (default 161)
    --n                     (default 1)

    --disp_range            (default 128)
    --savebin               (default 0)
    
    --postprocess           (default 0)
    --cost_agg              (default 2)
    --cost_agg_2            (default 2)
    --cost_w                (default 5)
    --cost_h                (default 5)

    --sgm                   (default 1)
    --post                  (default 1)
    --nyu_cost_agg_1        (default 2)
    --nyu_cost_agg_2        (default 0)
    --confLoc               (default 1)
    --thres                 (default 23)

    --small                 (default 0)
    --koi_sgm               (default 0)
    --koi_sps               (default 1)
    --unary_scale           (default 0)
]]
print(opt)

torch.manualSeed(123)
cutorch.setDevice(opt.gpuid+1)

if opt.data_version == 'kitti2015' then
    nChannel = 3
    left_folder = 'image_2'
    right_folder = 'image_3'
elseif opt.data_version == 'kitti2012' then
    nChannel = 1
    left_folder = 'image_0'
    right_folder = 'image_1'
else
    error('data_version should be either kitti2012 or kitti2015')
end

local function load_model()
    require('models/' .. opt.model .. '.lua')
    bottom, top = split_model(nChannel)
    bottom = bottom:cuda()
    top = top:cuda()

    local bottom_param, bottom_grad_param = bottom:getParameters()
    print(string.format('number of parameters: %d', bottom_param:nElement()))
    
    print(c.blue '==>' ..' loading parameters')
    -- load parameters
    local params = torch.load(opt.model_param)
    assert(params:nElement() == bottom_param:nElement(), string.format('%s: %d vs %d', 'loading parameters: dimension mismatch.', params:nElement(), bottom_param:nElement()))
    bottom_param:copy(params)

    if(string.len(opt.bn_meanstd) > 0) then 
        local bn_mean, bn_std = table.unpack(torch.load(opt.bn_meanstd))

        for k,v in pairs(bottom:findModules('nn.SpatialBatchNormalization')) do
            v.running_mean:copy(bn_mean[k])
            v.running_var:copy(bn_std[k])
        end
        bottom:evaluate() -- bn statistics required
        top:evaluate()
    end
end

function my_forward( m, img )    
    local data = img:view(1, img:size(1), img:size(2), img:size(3))
    local tic = torch.tic()
    for i = 1, #m do
        data = m.modules[i]:updateOutput(data)
        -- print('..' .. i)
        if m.modules[i].finput then
            m.modules[i].finput:set()
        end
    end
    print('feat comp tmr.. ' .. torch.toc(tic))
    return data:clone()
end

if string.len(opt.perm_fn) > 0 then
    file_ids = torch.FloatTensor(torch.FloatStorage(opt.perm_fn))
    print('load permutation: ' .. file_ids:size(1))
else
    file_ids = torch.range(0,199)
    print('use default order: 0-199')
end
paths.mkdir(opt.saveDir)


-- load model
print(c.blue '==>' ..' configuring model')
load_model()

for i = opt.start_id, opt.start_id+opt.n-1 do
    -- local file_id = file_ids[opt.val_id]
    local file_id = file_ids[i]
    print('--- ' ..  i .. ' --- ' .. ' fn: ' .. file_id)

    local l_fn = string.format('%s/%s/%06d_10.png', opt.data_root, left_folder, file_id)
    local r_fn = string.format('%s/%s/%06d_10.png', opt.data_root, right_folder, file_id)
    local l_img = image.load(l_fn, nChannel, 'byte'):cuda()
    local r_img = image.load(r_fn, nChannel, 'byte'):cuda()
    l_img:add(-l_img:mean()):div(l_img:std())
    r_img:add(-r_img:mean()):div(r_img:std())

    local img_h = l_img:size(2)
    local img_w = l_img:size(3)
    print('image size: ' .. img_h .. ' x ' .. img_w)

    if opt.small == 1 then
        -- return cuda tensor
        left_feat = bottom:forward(l_img:view(1, nChannel, img_h, img_w)):clone()
        right_feat = bottom:forward(r_img:view(1, nChannel, img_h, img_w)):clone()
    else
        left_feat = my_forward(bottom, l_img)
        right_feat = my_forward(bottom, r_img)
    end

    -- feature should be batch_sz x feat_dim x height x width
    -- print('feature dim: ')
    -- print(#left_feat)
    -- print()

    -- clean bottom memory
    -- clear_model()

    local total_loc = opt.disp_range
    unary_vol = torch.CudaTensor(img_h, img_w, total_loc):zero()
    right_unary_vol = torch.CudaTensor(img_h, img_w, total_loc):zero()

    local tic = torch.tic()

    local start_id, end_id = 1, opt.sub_width
    if end_id > img_w then end_id = img_w end

    -- general(use less memory, compute on sub-image and combine later) but potentially slower
    while start_id <= img_w do
        print(start_id, end_id, img_w)

        for loc_idx = 1, total_loc do
            local x_off = -loc_idx+1 -- always <= 0

            if end_id+x_off >= 1 and img_w >= start_id+x_off then
                --print(math.max(start_id,-x_off+1), math.min(end_id, img_w-x_off))
                local l = left_feat[{{}, {}, {}, {math.max(start_id,-x_off+1), math.min(end_id, img_w-x_off)}}]
                local r = right_feat[{{},{}, {}, {math.max(1,x_off+start_id), math.min(img_w,end_id+x_off)}}]

                local tmp = top:forward({l,r})
                unary_vol[{{},{math.max(start_id, -x_off+1), math.min(end_id, img_w-x_off)},loc_idx}]:copy(tmp)
                
                right_unary_vol[{{},{math.max(1,x_off+start_id), math.min(img_w,end_id+x_off)},loc_idx}]:copy(tmp)
                
            end
        end

        start_id = end_id + 1
        end_id = math.min(img_w, end_id+opt.sub_width)
    end
    print('matching time: ' .. torch.toc(tic))

    -- random output
    -- print('random output: ')
    -- print(unary_vol[{200,499,{51,55}}])
    if opt.savebin == 1 then
        torch.DiskFile(('%s/unary_vol_%06d.bin'):format(opt.saveDir, file_id), 'w'):binary():writeFloat(unary_vol:float():storage())
    end

    paths.mkdir(paths.concat(opt.saveDir, 'unary_img'))
    _,pred = unary_vol:max(3)
    pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
    image.save(string.format('%s/unary_img/%06d_10.png', opt.saveDir, file_id), pred:byte())

    paths.mkdir(paths.concat(opt.saveDir, 'right_unary_img'))
    _,right_pred = right_unary_vol:max(3)
    right_pred = right_pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
    image.save(string.format('%s/right_unary_img/%06d_10.png', opt.saveDir, file_id), right_pred:byte())
    -- unary_vol: h x w x disp
    print('writing unary image done..')

    if opt.postprocess == 1 then
        require 'smooth'
        if opt.cost_agg > 0 then
            print('cost agg..')
            local tic = torch.tic()
            cost_vol = unary_vol:permute(3,1,2):clone()
            local pad_w, pad_h = (opt.cost_w-1)/2, (opt.cost_h-1)/2
            local agg_model = nn.SpatialAveragePooling(opt.cost_w, opt.cost_h, 1, 1, pad_w, pad_h):cuda()
            agg_model:setCountExcludePad()

            for i = 1, opt.cost_agg do
                cost_vol = agg_model:forward(cost_vol):clone()
            end

            right_cost_vol = right_unary_vol:permute(3,1,2):clone()
            for i = 1, opt.cost_agg do
                right_cost_vol = agg_model:forward(right_cost_vol):clone()
            end
            print('cost agg tmr.. ' .. torch.toc(tic))

            paths.mkdir(paths.concat(opt.saveDir, 'cost_img'))
            score,pred = cost_vol:max(1)
            if opt.confLoc == 1 then
                pred[score:lt(opt.thres)] = 256
            end
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())

            paths.mkdir(paths.concat(opt.saveDir, 'right_cost_img'))
            _,pred = cost_vol:max(1)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/right_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
            -- cost_vol: disp x h x w
            print('writing cost image done..')
        end

        -------------------
        if opt.nyu_cost_agg_1 > 0 then
            print('nyu cost agg..')
            local tic = torch.tic()
            lu = unary_vol:view(1,img_h,img_w,opt.disp_range):permute(1,4,2,3):clone()
            ru = right_unary_vol:view(1,img_h,img_w,opt.disp_range):permute(1,4,2,3):clone()
            lu,ru = smooth.nyu.cross_agg(l_img:view(1,3,img_h,img_w), r_img:view(1,3,img_h,img_w), lu, ru, opt.nyu_cost_agg_1)
            print('nyu cost agg tmr.. ' .. torch.toc(tic))

            paths.mkdir(paths.concat(opt.saveDir, 'nyu_cost_img'))
            _,pred = lu:max(2)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/nyu_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())

            paths.mkdir(paths.concat(opt.saveDir, 'right_nyu_cost_img'))
            _,pred = ru:max(2)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/right_nyu_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
            print('writing NYU cost image done..')
        else
            lu = cost_vol:view(1, opt.disp_range, img_h, img_w)
            ru = right_cost_vol:view(1, opt.disp_range, img_h, img_w)
        end

        if opt.confLoc == 1 then
            lu[lu:lt(opt.thres)] = 0
            ru[ru:lt(opt.thres)] = 0
        end

        print('cost mean: ' .. c.cyan(lu:mean()) .. ' cost max: ' .. c.cyan(lu:max()) .. ' std: ' .. c.cyan(lu:std()))
        -- torch.save('debug.t7', lu:float())
        lu = lu / lu:std()
        ru = ru / ru:std()

        if opt.koi_sgm == 1 then
            print('koi sgm..')
            local tic = torch.tic()
            smooth.koi.sgm(l_fn, r_fn, lu, ru, opt.disp_range, opt.saveDir, opt.koi_sps, opt.unary_scale)
            print('koi sgm tmr.. ' .. torch.toc(tic))
            os.exit()
        end

        if opt.sgm == 1 then
            print('nyu sgm..')
            local tic = torch.tic()
            lu:mul(-1)
            ru:mul(-1)
            lu = smooth.nyu.sgm(l_img, r_img, lu, -1)
            ru = smooth.nyu.sgm(l_img, r_img, ru, 1)
            print('nyu sgm tmr.. ' .. torch.toc(tic))

            paths.mkdir(paths.concat(opt.saveDir, 'nyu_sgm_img'))
            _,pred = lu:min(2)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/nyu_sgm_img/%06d_10.png', opt.saveDir, file_id), pred:byte())

            paths.mkdir(paths.concat(opt.saveDir, 'right_nyu_sgm_img'))
            _,pred = ru:min(2)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/right_nyu_sgm_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
            -- lu: 1 x disp x h x w
            print('writing SGM image done..')
        end

        if opt.nyu_cost_agg_2 > 0 then
            print('nyu cost agg 2..')
            local tic = torch.tic()
            lu,ru = smooth.nyu.cross_agg(l_img:view(1,3,img_h,img_w), r_img:view(1,3,img_h,img_w), lu, ru, opt.nyu_cost_agg_2)
            print('nyu cost agg tmr.. ' .. torch.toc(tic))

            paths.mkdir(paths.concat(opt.saveDir, 'nyu_cost_img_2'))
            _,pred = lu:min(2)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/nyu_cost_img_2/%06d_10.png', opt.saveDir, file_id), pred:byte())

            paths.mkdir(paths.concat(opt.saveDir, 'right_nyu_cost_img_2'))
            _,pred = ru:min(2)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/right_nyu_cost_img_2/%06d_10.png', opt.saveDir, file_id), pred:byte())
        end

        if opt.cost_agg_2 > 0 then
            print('cost agg..')
            local tic = torch.tic()
            lu = lu:view(opt.disp_range, img_h, img_w)
            local pad_w, pad_h = (opt.cost_w-1)/2, (opt.cost_h-1)/2
            local agg_model = nn.SpatialAveragePooling(opt.cost_w, opt.cost_h, 1, 1, pad_w, pad_h):cuda()
            agg_model:setCountExcludePad()

            for i = 1, opt.cost_agg_2 do
                lu = agg_model:forward(lu):clone()
            end

            ru = ru:view(opt.disp_range, img_h, img_w)
            for i = 1, opt.cost_agg_2 do
                ru = agg_model:forward(ru):clone()
            end
            print('post cost agg tmr.. ' .. torch.toc(tic))

            paths.mkdir(paths.concat(opt.saveDir, 'post_cost_img'))
            _,pred = lu:min(1)
            -- if opt.confLoc == 1 then
            --     pred[score:lt(opt.thres)] = 256
            -- end
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/post_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())

            paths.mkdir(paths.concat(opt.saveDir, 'right_post_cost_img'))
            _,pred = ru:max(1)
            pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
            image.save(string.format('%s/right_post_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())

            lu = lu:view(1, opt.disp_range, img_h, img_w)
            ru = ru:view(1, opt.disp_range, img_h, img_w)
        end

        -- more nyu postprocess
        if opt.post == 1 then
            -- lu: 1 x disp x h x w
            disp = {}
            _, pred = lu:min(2)
            disp[1] = pred - 1
            _, pred = ru:min(2)
            disp[2] = pred - 1

            print('nyu post..')
            local tic = torch.tic()
            final_pred, outlier = smooth.nyu.post(disp, lu)
            print('nyu post tmr.. ' .. torch.toc(tic))

            paths.mkdir(paths.concat(opt.saveDir, 'nyu_post'))
            image.save(string.format('%s/nyu_post/%06d_10.png', opt.saveDir, file_id), final_pred:view(img_h, img_w):byte())

            paths.mkdir(paths.concat(opt.saveDir, 'outlier'))
            image.save(string.format('%s/outlier/%06d_10.png', opt.saveDir, file_id), (outlier*127):view(img_h, img_w):byte())
            print('writing NYU post image done..')
        end

        if opt.koi_sgm == 0 and opt.post == 1 and opt.koi_sps == 1 then
            -- use koi smooth only
            paths.mkdir(opt.saveDir..'/nyu_koi_final')
            local png_fn = string.format('%s/nyu_post/%06d_10.png', opt.saveDir, file_id)
            print('koi sps post..')
            local tic = torch.tic()
            smooth.koi.sps(l_fn, r_fn, png_fn, opt.saveDir..'/nyu_koi_final')
            print('koi sps post tmr.. ' .. torch.toc(tic))
            print('writing SPS post image done..')
        end
    end

end




