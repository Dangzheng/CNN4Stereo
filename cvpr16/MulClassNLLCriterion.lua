-- used with LogSoftMax

local MulClassNLLCriterion, parent = torch.class(
    'nn.MulClassNLLCriterion',
    'nn.Criterion'
)

function MulClassNLLCriterion:__init(gt_weight)
    parent.__init(self)

    if gt_weight then
        gt_weight:div(gt_weight:sum())
        self.gt_weight = gt_weight
    else
        self.gt_weight = torch.ones(1)
    end

    assert(self.gt_weight:nElement() % 2 == 1, 'nElement of gt_weight should be odd')
    self.half_width = (self.gt_weight:nElement() - 1)/ 2
    -- if func then
    --     self.func = func
    -- else
    --     self.func = function (x) return x end
    -- end

    -- self.output_tensor = torch.zeros(1)
    -- self.total_weight_tensor = torch.ones(1)
    -- self.target = torch.zeros(1):long()

    -- print(self.gt_weight)
end




function MulClassNLLCriterion:__len()
    return 0
end


function MulClassNLLCriterion:updateOutput(input, target)
    assert(type(target) ~= 'number', 'target should be a tensor')

    if target:type() == 'torch.CudaTensor' then
        self.target = target
    else
        self.target = target:long()
    end

    -- has dimension for batch-size
    assert(input:dim() == 2 and target:dim() == 2, 'input should be 2D')
    assert(target:size(2) == 1, string.format('only support 1 gt locaton, got: %d', target:size(2)))
    self.output = 0
    for i = 1,input:size(1) do
        local s,e = math.max(1,target[i][1]-self.half_width), math.min(input:size(2),target[i][1]+self.half_width)
        -- print(s,e)
        self.output = self.output - torch.cmul(input[{i,{s,e}}], self.gt_weight[{{self.half_width+1-(target[i][1]-s), self.half_width+1+e-target[i][1]}}]):sum()
    end

    return self.output
end

function MulClassNLLCriterion:updateGradInput(input, target)
    assert(type(target) ~= 'number', 'target should be a tensor')

    if target:type() == 'torch.CudaTensor' then
        self.target = target
    else
        self.target = target:long()
    end

    assert(input:dim() == 2, 'input should be 2D')
    self.gradInput:resizeAs(input):zero()
    
    for i = 1,input:size(1) do
        local s,e = math.max(1,target[i][1]-self.half_width), math.min(input:size(2),target[i][1]+self.half_width)
        self.gradInput[{i,{s,e}}]:copy(self.gt_weight[{{self.half_width+1-(target[i][1]-s), self.half_width+1+e-target[i][1]}}]):mul(-1)
    end

    return self.gradInput
end


