from evaluation import *

def test_model(model, test_loaders, save_pth):
    
    evaluator = Tester(test_loaders)
    
    meanDice = evaluator.eval(model)
    avg = np.round(np.mean(meanDice),4)
    std = np.round(np.std(meanDice),4)
        
    print('Total Average: {}, {}'.format(avg, std))

    np.save(save_pth+'results.npy', meanDice, allow_pickle=True)
        
    
class Tester(Evaluator):
    
    def eval(self, model):
        
        num_cls = 2
        total_overlap = np.zeros((1, num_cls, 5))
        res = {}
        model = copy.deepcopy(model).cpu()
        model.eval()
        for vali_batch in self.vali_loaders:

            #imgs = torch.from_numpy(vali_batch['data']).cuda(non_blocking=True)
            imgs = torch.from_numpy(vali_batch['data'])
            labs = vali_batch['seg']

            output= model(imgs)
 
            truemax, truearg0 = torch.max(output, 1, keepdim=False)

            truearg = truearg0.detach().cpu().numpy().astype(np.uint8)
            
            if len(labs.shape) == len(output.shape):
                labs = labs[:, 0]
            
            overlap_result, _ = Hausdorff_compute(truearg, labs, num_cls, (1.5,1.5,10,1))

            total_overlap = np.concatenate((total_overlap, overlap_result), axis=0)
            
            #del input, truearg0, truemax
        
        dice = total_overlap[1:,1, 1]
        
        return dice
