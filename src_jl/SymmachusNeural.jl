using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw

@with_kw mutable struct Args
    η::Float64 = 3e-4       # learning rate
    batchsize::Int = 1024   # batch size
    epochs::Int = 10        # number of epochs
end

function getdata(args)
    # Loading Dataset

    # Must be embedding array
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader(xtest, ytest, batchsize=args.batchsize)

    return train_data, test_data
end

function build_model(; embeddings_size=300)
    return Chain(
 	    Dense(embeddings_size, 100, relu),
            BatchNorm(100, relu),
            Dense(100, 25)
            BatchNorm(25),
            Dense(25, 1)
            )
end

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

function train(; kws...)
    # Initializing Model parameters
    args = Args(; kws...)

    # Load Data
    train_data,test_data = getdata(args)

    # Construct model
    m = build_model()
    loss(x,y) = logitcrossentropy(m(x), y)

    ## Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(args.η)

    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    @show accuracy(train_data, m)

    @show accuracy(test_data, m)
end
