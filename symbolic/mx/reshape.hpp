/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010 by Joel Andersson, Moritz Diehl, K.U.Leuven. All rights reserved.
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "mx_node.hpp"
#include <map>
#include <stack>

namespace CasADi{
  /** \brief Reshape an expression
      \author Joel Andersson
      \date 2013
  */
  class Reshape : public MXNode{
  public:

    /// Constructor
    Reshape(const MX& x, CRSSparsity sp);

    /// Clone function
    virtual Reshape* clone() const;
      
    /// Destructor
    virtual ~Reshape(){}
    
    /// Evaluate the function numerically
    virtual void evaluateD(const DMatrixPtrV& input, DMatrixPtrV& output, const DMatrixPtrVV& fwdSeed, DMatrixPtrVV& fwdSens, const DMatrixPtrVV& adjSeed, DMatrixPtrVV& adjSens);

    /// Evaluate the function symbolically (SX)
    virtual void evaluateSX(const SXMatrixPtrV& input, SXMatrixPtrV& output, const SXMatrixPtrVV& fwdSeed, SXMatrixPtrVV& fwdSens, const SXMatrixPtrVV& adjSeed, SXMatrixPtrVV& adjSens);

    /// Evaluate the function symbolically (MX)
    virtual void evaluateMX(const MXPtrV& input, MXPtrV& output, const MXPtrVV& fwdSeed, MXPtrVV& fwdSens, const MXPtrVV& adjSeed, MXPtrVV& adjSens, bool output_given);

    /// Propagate sparsity
    virtual void propagateSparsity(DMatrixPtrV& input, DMatrixPtrV& output, bool fwd);

    /// Print a part of the expression */
    virtual void printPart(std::ostream &stream, int part) const;
    
    /** \brief Generate code for the operation */
    virtual void generateOperation(std::ostream &stream, const std::vector<std::string>& arg, const std::vector<std::string>& res, CodeGenerator& gen) const;

    /// Evaluate the function (template)
    template<typename T, typename MatV, typename MatVV> 
    void evaluateGen(const MatV& input, MatV& output, const MatVV& fwdSeed, MatVV& fwdSens, const MatVV& adjSeed, MatVV& adjSens);    
    
    /** \brief Get the operation */
    virtual int getOp() const{ return OP_RESHAPE;}

    /// Can the operation be performed inplace (i.e. overwrite the result)
    virtual bool allowInplace() const{ return false;}

  };

} // namespace CasADi

#endif // RESHAPE_HPP
