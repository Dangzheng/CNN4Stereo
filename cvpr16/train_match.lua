--- train with 3 pixel weighted log-loss 
--
-- Wenjie Luo
--

require 'xlua'
require 'optim'
require 'cunn'
require 'gnuplot'

require 'MulClassNLLCriterion.lua'
require('DataHandler.lua')
local c = require 'trepl.colorize'

opt = lapp[[
    --data_version             (default "kitti2015")
    -s,--save                  (default "logs/debug")       directory to save logs
    -b,--batchSize             (default 128)                batch size
    -g, --gpuid                (default 0)                  gpu id
    --tr_num                   (default 10)                 training images
    --val_num                  (default 1)                  validation images
    -r,--learningRate          (default 1e-2)               learning rate
    --learningRateDecay        (default 1e-7)               learning rate decay
    --weightDecay              (default 0.0005)             weightDecay
    -m,--momentum              (default 0.9)                momentum
    --model                    (default dot_win37_dep9)     model name
    --epoch_step               (default 40)                 half learning rate at every 'epoch_step'
    --weight_epoch             (default 5)                  save weight at every 'weight_epoch'
    --max_epoch                (default 10)                 maximum number of iterations
    --iter_per_epoch           (default 50)                 evaluate every # iterations, and update plot
    --data_root                (default "/ais/gobi3/datasets/kitti/scene_flow/training") dataset root folder
    --util_root                (default "")                 dataset root folder
    --tb                       (default 100)                test batch size
    --num_val_loc              (default 10000)              number test patch pair
    --opt_method               (default 'adam')             sgd, adagrad, adam

    --showCurve                (default 0)                  use 1 to show training / validation curve
    --psz                      (default 9)                  half width
    --half_range               (default 100)                half range
]]

print(opt)

print(c.blue '==>' ..' configuring model')

torch.manualSeed(123)
cutorch.setDevice(opt.gpuid+1)
torch.setdefaulttensortype('torch.FloatTensor')

print(c.blue '==>' ..' loading data')
my_dataset = DataHandler(opt.data_version, opt.data_root, opt.util_root, opt.tr_num, opt.val_num, opt.num_val_loc, opt.batchSize, opt.psz, opt.half_range, 1)

print(c.blue '==>' ..' create model')
require('models/' .. opt.model .. '.lua')
if opt.data_version == 'kitti2015' then
    model = create_model(opt.half_range*2 + 1, 3):cuda()
elseif opt.data_version == 'kitti2012' then
    model = create_model(opt.half_range*2 + 1, 1):cuda()
else
    error('data_version should be either kitti2012 or kitti2015')
end
parameters,gradParameters = model:getParameters()
print(string.format('number of parameters: %d', parameters:nElement()))

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean patch pred accuracy (train set)', '% mean patch pred accuracy (test set)'}
if opt.showCurve == 0 then
    testLogger.showPlot = false
end

print(c.blue'==>' ..' setting criterion')
gt_weight = torch.Tensor({1,4,10,4,1})--torch.ones(7)
criterion = nn.MulClassNLLCriterion(gt_weight):cuda()
--criterion = nn.ClassNLLCriterion():cuda()

print(c.blue'==>' ..' configuring optimizer')
optimConfig = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}
optimState = {}
optimMethod = optim[opt.opt_method]

epoch = epoch or 1
function train()
    local acc_count = 0
    model:training()
    -- drop learning rate every "epoch_step" epochs
    if epoch == 120 then optimConfig.learningRate = optimConfig.learningRate/5 end
    if epoch > 120 and (epoch - 120) % opt.epoch_step == 0 then optimConfig.learningRate = optimConfig.learningRate/5 end

    print(c.blue '==>'.." fake epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local tic = torch.tic()
    local train_loss = -1
    for t = 1, opt.iter_per_epoch do
        xlua.progress(t, opt.iter_per_epoch)

        local left, right, targets = my_dataset:next_batch()
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()
            local outputs = model:forward({left, right})
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward({left, right}, df_do)

            -- 3 pixel error
            local _,y = outputs:max(2)
            acc_count = acc_count + (torch.abs(y-targets):le(3):sum())

            return f,gradParameters
        end
        local _, loss = optimMethod(feval, parameters, optimConfig, optimState)
        train_loss = loss[1]
    end

    train_acc = acc_count/(opt.iter_per_epoch*opt.batchSize) * 100
    print(('Train loss: '..c.cyan'%.4f' .. ' train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s\t grad/param norm = %6.4e\t learning rate: %f'):format(train_loss/opt.batchSize, train_acc, torch.toc(tic), gradParameters:norm() / parameters:norm(), optimConfig.learningRate))

    epoch = epoch + 1
end

acc_count = 0
function evaluate()
    -- compute 3-pixel error
    model:evaluate()
    print(c.blue '==>'.." validation")
    local l, r, tar = my_dataset:get_eval_cuda()

    -- tar:mul(-1):add(opt.max_disp+1)

    local n = (#l)[1]
    assert(math.fmod(n, opt.tb) == 0, "use opt.tb to be divided exactly by number of validate sample")
    acc_count = 0
    for i=1,n,opt.tb do
        _,y = model:forward({l:narrow(1,i,opt.tb), r:narrow(1,i,opt.tb)}):max(2)
        acc_count = acc_count + (torch.abs(y-tar:narrow(1,i,opt.tb)):le(3):sum())
        -- print(y, tar)
    end

    acc_count = acc_count / n * 100
    print('Test accuracy: ' .. c.cyan(acc_count) .. ' %')

    if epoch % 10 == 0 then collectgarbage() end
  -- confusion:zero()
end

function logging( )
    
    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add{train_acc, acc_count}
        testLogger:style{'-','-'}
        testLogger:plot()
        
        os.execute('convert -density 200 '..opt.save..'/test.log.eps '..opt.save..'/test.png')
    end

    -- save model parameters every # epochs
    if epoch % opt.weight_epoch == 0 or epoch == opt.max_epoch then
        local filename = paths.concat(opt.save, string.format('param_epoch_%d.t7', epoch))
        print('==> saving parameters to '..filename)
        torch.save(filename, parameters)

        -- save bn statistics from training set
        filename = paths.concat(opt.save, string.format('bn_meanvar_epoch_%d.t7', epoch))
        print('==> saving bn mean var to '..filename)
        local bn_mean = {}
        local bn_var = {}
        for k,v in pairs(model:findModules('nn.SpatialBatchNormalization')) do
            bn_mean[k] = v.running_mean
            bn_var[k] = v.running_var
        end
        if #bn_mean > 0 then torch.save(filename, {bn_mean, bn_var}) end
    end
end

while epoch < opt.max_epoch do
    train()

    if opt.num_val_loc > 0 then
        evaluate()
    end

    logging()
end


