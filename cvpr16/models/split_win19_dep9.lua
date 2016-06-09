-- trained using one patches, with dot product
require 'nn'
require 'cunn'

function split_model(nChannel)

	local half_padding = 9
	local bottom = nn.Sequential()
	bottom:add(nn.SpatialReflectionPadding(half_padding, half_padding, half_padding, half_padding))

	-- building block
	local function ConvBNReLU(nInputPlane, nOutputPlane, kw, kh, pw, ph)
	  pw = pw or 0
	  ph = ph or 0
	  bottom:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kw, kh, 1, 1, pw, ph))
	  bottom:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
	  bottom:add(nn.ReLU(true))
	  return bottom
	end

	ConvBNReLU( nChannel,64,3,3)
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.5))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))
	ConvBNReLU(64,64,3,3)--:add(nn.Dropout(0.4))

	-- ConvBNReLU(64,64,3,3,1,1)--:add(nn.Dropout(0.4))
	bottom:add(nn.SpatialConvolution(64, 64, 3, 3))
	bottom:add(nn.SpatialBatchNormalization(64,1e-3))


	-- left: bz * feat_dim * h * w

	local top = nn.Sequential()
	top:add(nn.CMulTable())
	top:add(nn.Sum(2))

	return bottom, top

end