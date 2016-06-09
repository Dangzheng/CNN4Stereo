-- trained using one patches, with dot product
-- difference with v10: log softmax

require 'nn'
require 'cunn'

function create_model(max_dips, nChannel)
	local m = nn.Sequential()

	local bottom = nn.ParallelTable()
	local bottom_left = nn.Sequential()
	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane, kw, kh)
	  bottom_left:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kw, kh))
	  bottom_left:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
	  bottom_left:add(nn.ReLU(true))
	  return bottom_left
	end

	local c = nChannel or 3
	ConvBNReLU(c,64,3,3)
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	-- ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	bottom_left:add(nn.SpatialConvolution(64, 64, 3, 3))
	bottom_left:add(nn.SpatialBatchNormalization(64,1e-3))	-- left: bz * feat_dim * 1 * 1
	-- right: bz * feat_dim * 1 * (1+256)

	local bottom_right = bottom_left:clone('weight','bias','gradWeight','gradBias')
	bottom_left:add(nn.Transpose({2,3}, {3,4}))
	bottom_left:add(nn.Reshape(1, 64))
	-- bz * (1 * feat_dim)

	bottom_right:add(nn.Reshape(64,max_dips))
	-- bz * (feat_dim * 1+256)

	bottom:add(bottom_left):add(bottom_right)

	m:add(bottom)
	m:add(nn.MM())
	-- bz * 1 * (1+256)
	m:add(nn.Reshape(max_dips))
	-- bz * (1+256)

	m:add(nn.LogSoftMax())
	-- a = torch.rand(10,1,9,9)
	-- b = torch.rand(10,1,9,9+256)
	-- print(m:forward({a,b}))

	return m

end