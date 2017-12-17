#include "layers/generic.h"
#include <typeinfo>


namespace marian {

Expr Cost(Expr logits,
          Expr indices,
          Expr mask,
          std::string costType,
          float smoothing) {
  using namespace keywords;

  auto ce = cross_entropy(logits, indices);

  if(smoothing > 0) {
    // @TODO: add this to CE kernels instead
    auto ceq = mean(logsoftmax(logits), axis = -1);
    ce = (1 - smoothing) * ce - smoothing * ceq;
  }

  if(mask)
    ce = ce * mask;

  Expr cost;
  if(costType == "ce-mean" || costType == "cross-entropy") {
    cost = mean(sum(ce, axis = -3), axis = -2);
  } else if(costType == "ce-mean-words") {
    cost
        = sum(sum(ce, axis = -3), axis = -2) / sum(sum(mask, axis = -3), axis = -2);
  } else if(costType == "ce-sum") {
    cost = sum(sum(ce, axis = -3), axis = -2);
  } else if(costType == "perplexity") {
    cost = exp(sum(sum(ce, axis = -3), axis = -2)
               / sum(sum(mask, axis = -3), axis = -2));
  } else if(costType == "ce-rescore") {
    cost = -sum(ce, axis = -3);
  } else {  // same as ce-mean
    cost = mean(sum(ce, axis = -3), axis = -2);
  }


  // HK if we have a weight, use it here? 
  //todo: check if this is per sentence / per words
  std::cout << "hk cost is: "<< cost << "\n";
  std::cout << "hk cost type: "<< typeid(cost).name()<< "\n\n";

  return cost;
}
}
