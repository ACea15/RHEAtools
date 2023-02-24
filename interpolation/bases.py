
import numpy as np
import itertools
import functools
import itertools

class RadialBases:
    """Documentation for RadialBasis

    """
    def __init__(self, *args, **kwargs): pass

    def rb_multiquadric(self, points, a=1, b=1., c=1., out_inputs=False):
        """
        'Multiquadric': a*sqrt((r/b)**2 + c)
        Interpolation: sqrt((r/epsilon)**2 + 1)
        """
        
        a, b, c = self.float2vect(len(points), a, b, c)
        if out_inputs:
            return [a,b,c]
    
        bases = []
        for ix, v in enumerate(points):
            z = lambda x,v=v: a[ix]*np.sqrt((np.linalg.norm([x[i]
                -vi for i, vi in enumerate(v)])/b[ix])**2 + c[ix])
            bases.append(z)
        return bases

    def rb_gaussian(self, points, a=1, b=1., c=0., out_inputs=False):
        """
        'Gaussian': a*exp(-(r/b + c)**2)
        Interpolation: exp(-(r/epsilon)**2)
        """

        a, b, c = self.float2vect(len(points), a, b, c)
        if out_inputs:
            return [a,b,c]

        bases = []
        for ix, v in enumerate(points):
            z = lambda x,v=v: a[ix]*np.exp(-(np.linalg.norm([x[i]
                    -vi for i, vi in enumerate(v)])/b[ix] + c[ix])**2)
            bases.append(z)
        return bases
    
    def rb_inverse(self, points, a=1, b=1., c=1., out_inputs=False):
        """
        'Inverse': a*1./sqrt((r/b)**2 +c)
        Interpolation: 1./sqrt((r/epsilon)**2+1)
        """

        a, b, c = self.float2vect(len(points), a, b, c)
        if out_inputs:
            return [a, b, c]

        bases = []
        for ix, v in enumerate(points):
            z = lambda x,v=v: a[ix]/np.sqrt((np.linalg.norm([x[i]
                -vi for i, vi in enumerate(v)])/b[ix])**2 + c[ix])
            bases.append(z)
        return bases

    def rb_linear(self, points, a=1., b=0., c=0., out_inputs=False):
        """
        'Linear': a*r +c
        Interpolation: r
        """

        a, b, c = self.float2vect(len(points), a, b, c)
        if out_inputs:
            return [a, b, c]

        bases = []
        for ix, v in enumerate(points):
            z = lambda x,v=v: a[ix]*np.linalg.norm([x[i]
                -vi for i, vi in enumerate(v)]) + c[ix]
            bases.append(z)
        return bases

    @staticmethod
    def float2vect(args_len,*args):
        """
        Converts every number in args into a constant vector of length args_len
        """
        argx = [[] for i in range(len(args))]
        for ai, av in enumerate(args):
            if isinstance(av, float) or isinstance(av, int):
                argx[ai] = [av]*args_len
            else:
                assert len(av)==args_len, "mismatch in length of inputs to radial bases"
                argx[ai] = av
        return argx   
    
class PolynomialBases:
    """
    Documentation for PolynomialBasis
    """
    def __init__(self, *args, **kwargs): pass

    def pb_selective(self, combination, degree, coeff=1.):

        poly_ind = list(itertools.combinations_with_replacement(combination,
                                                                degree))
        if isinstance(coeff,float) or isinstance(coeff,int):
            coeff = [coeff]*len(poly_ind)

        bases = []
        for i_p, v_p in enumerate(poly_ind):
            base_i = [j-1 for j in v_p if j > 0]
            bases.append(lambda x:coeff[i_p]*np.prod([x[k] for k in base_i]))

        return bases

    def poly_bases(self, dim, degree, coeff=1.):

        poly_ind = list(itertools.combinations_with_replacement(range(dim+1),
                                                                degree))
        if isinstance(coeff, float) or isinstance(coeff, int):
            coeff = [coeff]*len(poly_ind)

        bases = []
        for i_p, v_p in enumerate(poly_ind):
            base_i = [j-1 for j in v_p if j > 0]
            bases.append(lambda x, base_i=base_i: coeff[i_p]*np.prod([x[k] for k in base_i]))

        return bases

    def poly_value(self, X, bases):

        num_bases = len(bases)
        value = np.sum([bases[i](X) for i in range(num_bases)])
        return value
    
    def pb_polyfull(self, points, dim, degree, coeff, out_inputs=False):

        """
        'Linear': a*r +c
        Interpolation: r
        """
        if out_inputs:
            return [np.zeros(len(list(itertools.combinations_with_replacement(
                            range(dim+1),degree)))) for i in range(len(points))]
        bases = []
        for ix, v in enumerate(points):
            coeff_i = coeff[ix]
            z = lambda x,coeff_i=coeff_i: self.poly_value(x,
                                                          self.poly_bases(dim,
                                                                          degree,
                                                                          coeff_i)
                                                          )
            bases.append(z)
        return bases

#scipy.spatial.distance.cdist
class TrigoBases:
    """Documentation for PolynomialBasis

    """
    def __init__(self, *args, **kwargs): pass


class Bases(RadialBases, PolynomialBases, TrigoBases):

    def __init__(self, *args, **kwargs): pass

