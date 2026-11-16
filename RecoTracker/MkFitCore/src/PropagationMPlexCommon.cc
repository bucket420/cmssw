#include "FWCore/Utilities/interface/CMSUnrollLoop.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "PropagationMPlex.h"

#include <vdt/log.h>
#include <vdt/sincos.h>

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  // this version does not assume to know which elements are 0 or 1, so it does the full multiplication
  void MultHelixPropFull(const MPlexLL& A, const MPlexLS& B, MPlexLL& C) {
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
// optimization reports indicate only the inner two loops are good
// candidates for vectorization
#pragma omp simd
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += A.constAt(n, i, k) * B.constAt(n, k, j);
        }
      }
    }
  }

  // this version does not assume to know which elements are 0 or 1, so it does the full mupltiplication
  void MultHelixPropTranspFull(const MPlexLL& A, const MPlexLL& B, MPlexLS& C) {
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
// optimization reports indicate only the inner two loops are good
// candidates for vectorization
#pragma omp simd
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += B.constAt(n, i, k) * A.constAt(n, j, k);
        }
      }
    }
  }

#ifdef UNUSED
  // this version does not assume to know which elements are 0 or 1, so it does the full multiplication
  void MultHelixPropFull(const MPlexLL& A, const MPlexLL& B, MPlexLL& C) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += A.constAt(n, i, k) * B.constAt(n, k, j);
        }
      }
    }
  }

  // this version does not assume to know which elements are 0 or 1, so it does the full mupltiplication
  void MultHelixPropTranspFull(const MPlexLL& A, const MPlexLL& B, MPlexLL& C) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
          C(n, i, j) = 0.;
          for (int k = 0; k < 6; ++k)
            C(n, i, j) += B.constAt(n, i, k) * A.constAt(n, j, k);
        }
      }
    }
  }
#endif

  //==============================================================================

  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            const MPlexHV& plNrm,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            const int N_proc,
                            const float* msRefP) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      if (n >= N_proc)
        continue;
      float radL = hitsRl.constAt(n, 0, 0);
      if (radL < 1e-13f)
        continue;  //ugly, please fixme
      const float theta = outPar.constAt(n, 5, 0);
      // const float pt = 1.f / outPar.constAt(n, 3, 0);  //fixme, make sure it is positive?
      const float ipt = outPar.constAt(n, 3, 0);
      const float pt = 1.f / ipt;  //fixme, make sure it is positive?
      const float ipt2 = ipt * ipt;
      float sT;
      float cT;
      vdt::fast_sincosf(theta, sT, cT);
      const float p = pt / sT;
      const float pz = p * cT;
      const float p2 = p * p;
      constexpr float mpi = 0.140;       // m=140 MeV, pion
      constexpr float mpi2 = mpi * mpi;  // m=140 MeV, pion
      const float beta2 = p2 / (p2 + mpi2);
      const float beta = std::sqrt(beta2);
      //radiation lenght, corrected for the crossing angle (cos alpha from dot product of radius vector and momentum)
      float sinP;
      float cosP;
      vdt::fast_sincosf(outPar.constAt(n, 4, 0), sinP, cosP);
      const float invCos = p / std::abs(pt * cosP * plNrm.constAt(n, 0, 0) + pt * sinP * plNrm.constAt(n, 1, 0) +
                                        pz * plNrm.constAt(n, 2, 0));
      radL = radL * invCos;  // general: invCos is p/|p.n| with n the module normal
      // multiple scattering
      //vary independently phi and theta by the rms of the planar multiple scattering angle
      // XXX-KMD radL < 0, see your fixme above! Repeating bailout
      if (radL < 1e-13f)
        continue;
      // const float thetaMSC = 0.0136f*std::sqrt(radL)*(1.f+0.038f*vdt::fast_logf(radL))/(beta*p);// eq 32.15
      // const float thetaMSC2 = thetaMSC*thetaMSC;
      // msRefP: theta0 at a fixed reference momentum instead of the running estimate.  Noise evaluated at the
      // running estimate correlates the assumed noise with the estimate's own error, which biases the
      // curvature of a scattering-dominated fit high (pT low).  Only theta0 changes: the geometric factors
      // below still use the current state, and energy loss is untouched.
      const float pMS = msRefP ? msRefP[n] : p;
      const float betaMS = msRefP ? std::sqrt(pMS * pMS / (pMS * pMS + mpi2)) : beta;
      const float thetaMSC = 0.0136f * (1.f + 0.038f * vdt::fast_logf(radL)) / (betaMS * pMS);  // eq 32.15
      const float thetaMSC2 = thetaMSC * thetaMSC * radL;
      if /*constexpr*/ (Config::usePtMultScat) {
        outErr.At(n, 3, 3) += thetaMSC2 * pz * pz * ipt2 * ipt2;
        outErr.At(n, 3, 5) -= thetaMSC2 * pz * ipt2;
        outErr.At(n, 4, 4) += thetaMSC2 * p2 * ipt2;
        outErr.At(n, 5, 5) += thetaMSC2;
      } else {
        outErr.At(n, 4, 4) += thetaMSC2;
        outErr.At(n, 5, 5) += thetaMSC2;
      }
      //std::cout << "beta=" << beta << " p=" << p << std::endl;
      //std::cout << "multiple scattering thetaMSC=" << thetaMSC << " thetaMSC2=" << thetaMSC2 << " radL=" << radL << std::endl;
      //std::cout << "radL=" << hitsRl.constAt(n, 0, 0) << " beta=" << beta << " invCos=" << invCos << " radLCorr=" << radL << " thetaMSC=" << thetaMSC << " thetaMSC2=" << thetaMSC2 << std::endl;
      // energy loss
      // XXX-KMD beta2 = 1 => 1 / sqrt(0)
      // const float gamma = 1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      // const float gamma2 = gamma*gamma;
      const float gamma2 = (p2 + mpi2) / mpi2;
      const float gamma = std::sqrt(gamma2);  //1.f/std::sqrt(1.f - std::min(beta2, 0.999999f));
      constexpr float me = 0.0005;            // m=0.5 MeV, electron
      const float wmax = 2.f * me * beta2 * gamma2 / (1.f + 2.f * gamma * me / mpi + me * me / (mpi * mpi));
      constexpr float I = 16.0e-9 * 10.75;
      const float deltahalf = vdt::fast_logf(28.816e-9f * std::sqrt(2.33f * 0.498f) / I) - 0.5f;
      // true formula below, but simplifying log(beta*gamma)
      // vdt::fast_logf(28.816e-9f * std::sqrt(2.33f * 0.498f) / I) + vdt::fast_logf(beta * gamma) - 0.5f;

      const float dEdx = beta < 1.f
                             ? (2.f * (hitsXi.constAt(n, 0, 0) * invCos *
                                       (0.5f * vdt::fast_logf(2.f * me * wmax / (I * I)) - beta2 - deltahalf) / beta2))
                             : 0.f;  //protect against infs and nans
      // true formula below, but simplifying 0.5*log(beta2*gamma2)
      // (2.f * (hitsXi.constAt(n, 0, 0) * invCos * (0.5f * vdt::fast_logf(2.f * me * beta2 * gamma2 * wmax / (I * I)) - beta2 - deltahalf) / beta2))

      // dEdx = dEdx*2.;//xi in cmssw is defined with an extra factor 0.5 with respect to formula 27.1 in pdg
      //std::cout << "dEdx=" << dEdx << " delta=" << deltahalf << " wmax=" << wmax << " Xi=" << hitsXi.constAt(n,0,0) << std::endl;
      const float dP = propSign.constAt(n, 0, 0) * dEdx / beta;
      outPar.At(n, 3, 0) = p / (std::max(p - dP, 0.001f) * pt);  //stay above 1MeV
      // Bethe-Bloch straggling variance, the CMS EnergyLossUpdator form.
      const float dEdx2 = (hitsXi.constAt(n, 0, 0) * invCos / beta2) * wmax * (1 - beta2 * 0.5);
      outErr.At(n, 3, 3) += dEdx2 / (beta2 * p2 * pt * pt);
    }
  }

  //==============================================================================
  // Curvilinear error helpers
  //==============================================================================

  namespace {
    // Non-zero patterns of the 5x5 jacobians acting on curvilinear errors.
    // Transport in curvilinear coordinates (errorPropCurv in PropagationMPlexPlane.cc)
    constexpr int kNZCurvProp[5][5] = {
        {1, 0, 0, 0, 0}, {0, 1, 1, 0, 0}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    // jacCurv2Loc in plane::to_local()
    constexpr int kNZCurv2Loc[5][5] = {
        {1, 0, 0, 0, 0}, {0, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}};
    // jacLoc2Curv in plane::to_global()
    constexpr int kNZLoc2Curv[5][5] = {
        {1, 0, 0, 0, 0}, {0, 1, 1, 0, 0}, {0, 1, 1, 1, 1}, {0, 0, 0, 1, 1}, {0, 0, 0, 1, 1}};

    // C = A * B * A^T, skipping the structural zeros of A; C may alias B
    template <const int (&NZ)[5][5]>
    void similarityCurvSparse(const MPlex55& A, const MPlex5S& B, MPlex5S& C) {
      MPlex55 temp;
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
#pragma GCC unroll 5
        for (int i = 0; i < 5; ++i) {
#pragma GCC unroll 5
          for (int j = 0; j < 5; ++j) {
            float sum = 0.f;
#pragma GCC unroll 5
            for (int k = 0; k < 5; ++k)
              if (NZ[i][k])
                sum += A.constAt(n, i, k) * B.constAt(n, k, j);
            temp(n, i, j) = sum;
          }
        }
      }
#pragma omp simd
      for (int n = 0; n < NN; ++n) {
#pragma GCC unroll 5
        for (int i = 0; i < 5; ++i) {
#pragma GCC unroll 5
          for (int j = 0; j <= i; ++j) {
            float sum = 0.f;
#pragma GCC unroll 5
            for (int k = 0; k < 5; ++k)
              if (NZ[j][k])
                sum += temp.constAt(n, i, k) * A.constAt(n, j, k);
            C(n, i, j) = sum;
          }
        }
      }
    }
  }  // namespace

  void similarityCurvProp(const MPlex55& A, const MPlex5S& B, MPlex5S& C) {
    similarityCurvSparse<kNZCurvProp>(A, B, C);
  }
  void similarityCurv2Loc(const MPlex55& A, const MPlex5S& B, MPlex5S& C) {
    similarityCurvSparse<kNZCurv2Loc>(A, B, C);
  }
  void similarityLoc2Curv(const MPlex55& A, const MPlex5S& B, MPlex5S& C) {
    similarityCurvSparse<kNZLoc2Curv>(A, B, C);
  }

  // Jacobian from CCS (x, y, z, 1/pT, phi, theta) to curvilinear (q/p, lambda, phi, x_perp, y_perp),
  // see TrackState::jacobianCCSToCurvilinear
  void jacobianCCSToCurv(const MPlexLV& par, const MPlexQI& chg, MPlex56& jac) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;
    MPF sinP, cosP;
    mpt::fast_sincos(par(4, 0), sinP, cosP);
    MPF sinT, cosT;
    mpt::fast_sincos(par(5, 0), sinT, cosT);
    jac.setVal(0.f);
    jac.aij(0, 3) = mpt::negate_if_ltz(sinT, chg);
    jac.aij(0, 5) = mpt::negate_if_ltz(cosT * par(3, 0), chg);
    jac.aij(1, 5) = -1.f;
    jac.aij(2, 4) = 1.f;
    jac.aij(3, 0) = -sinP;
    jac.aij(3, 1) = cosP;
    jac.aij(4, 0) = -cosP * cosT;
    jac.aij(4, 1) = -sinP * cosT;
    jac.aij(4, 2) = sinT;
  }

  // Jacobian from curvilinear to CCS, see TrackState::jacobianCurvilinearToCCS
  void jacobianCurvToCCS(const MPlexLV& par, const MPlexQI& chg, MPlex65& jac) {
    namespace mpt = Matriplex;
    using MPF = MPlexQF;
    MPF sinP, cosP;
    mpt::fast_sincos(par(4, 0), sinP, cosP);
    MPF sinT, cosT;
    mpt::fast_sincos(par(5, 0), sinT, cosT);
    jac.setVal(0.f);
    jac.aij(0, 3) = -sinP;
    jac.aij(0, 4) = -cosT * cosP;
    jac.aij(1, 3) = cosP;
    jac.aij(1, 4) = -cosT * sinP;
    jac.aij(2, 4) = sinT;
    jac.aij(3, 0) = mpt::negate_if_ltz(1.f / sinT, chg);
    jac.aij(3, 1) = par(3, 0) * cosT / sinT;
    jac.aij(4, 2) = 1.f;
    jac.aij(5, 1) = -1.f;
  }

  void convertErrCCSToCurv(const MPlexLS& inErr, const MPlexLV& par, const MPlexQI& chg, MPlex5S& outErr) {
    MPlex56 jac;
    jacobianCCSToCurv(par, chg, jac);
    MPlex56 temp;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 6; ++j) {
          float sum = 0.f;
          for (int k = 0; k < 6; ++k)
            sum += jac.constAt(n, i, k) * inErr.constAt(n, k, j);
          temp(n, i, j) = sum;
        }
      }
    }
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 5; ++i) {
        for (int j = 0; j <= i; ++j) {
          float sum = 0.f;
          for (int k = 0; k < 6; ++k)
            sum += temp.constAt(n, i, k) * jac.constAt(n, j, k);
          outErr(n, i, j) = sum;
        }
      }
    }
  }

  void convertErrCurvToCCS(const MPlex5S& inErr, const MPlexLV& par, const MPlexQI& chg, MPlexLS& outErr) {
    MPlex65 jac;
    jacobianCurvToCCS(par, chg, jac);
    MPlex65 temp;
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j) {
          float sum = 0.f;
          for (int k = 0; k < 5; ++k)
            sum += jac.constAt(n, i, k) * inErr.constAt(n, k, j);
          temp(n, i, j) = sum;
        }
      }
    }
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      for (int i = 0; i < 6; ++i) {
        for (int j = 0; j <= i; ++j) {
          float sum = 0.f;
          for (int k = 0; k < 5; ++k)
            sum += temp.constAt(n, i, k) * jac.constAt(n, j, k);
          outErr(n, i, j) = sum;
        }
      }
    }
  }

  // Material effects on a curvilinear error.
  // The CCS terms of applyMaterialEffects() are projected to curvilinear with the CCS->curvilinear
  // jacobian at the post-energy-loss state, reproducing what the CCS-error fit gets when the update
  // converts the propagated CCS error to curvilinear.
  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            const MPlexHV& plNrm,
                            const MPlexQI& inChg,
                            MPlex5S& outErr,
                            MPlexLV& outPar,
                            const int N_proc,
                            const float* msRefP) {
    MPlexLS matErr(0.f);
    MPlexQF iptPre;
#pragma omp simd
    for (int n = 0; n < NN; ++n)
      iptPre(n, 0, 0) = outPar(n, 3, 0);

    applyMaterialEffects(hitsRl, hitsXi, propSign, plNrm, matErr, outPar, N_proc, msRefP);

#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      if (n >= N_proc)
        continue;
      const float q = inChg(n, 0, 0);
      const float sT = std::sin(outPar(n, 5, 0));
      const float cT = std::cos(outPar(n, 5, 0));
      const float ipt = outPar(n, 3, 0);

      // The CCS-error fit transports with the curvilinear->CCS jacobian at the pre-energy-loss 1/pT
      // and converts back at the post-energy-loss one: their product is the identity except
      // d(q/p)/d(lambda) = q*cos(theta)*(ipt_pre - ipt_post).
      const float e = q * cT * (iptPre(n, 0, 0) - ipt);
      if (e != 0.f) {
        const float c01 = outErr(n, 0, 1);
        const float c11 = outErr(n, 1, 1);
        outErr(n, 0, 0) += e * (2.f * c01 + e * c11);
        outErr(n, 0, 1) += e * c11;
        outErr(n, 0, 2) += e * outErr(n, 1, 2);
        outErr(n, 0, 3) += e * outErr(n, 1, 3);
        outErr(n, 0, 4) += e * outErr(n, 1, 4);
      }

      // CCS -> curvilinear: q/p row (q*sT, q*cT*ipt) on (1/pT, theta), lambda = -theta, phi = phi
      const float a = q * sT;
      const float b = q * cT * ipt;
      const float m33 = matErr(n, 3, 3);
      const float m35 = matErr(n, 3, 5);
      const float m44 = matErr(n, 4, 4);
      const float m55 = matErr(n, 5, 5);
      outErr(n, 0, 0) += a * a * m33 + 2.f * a * b * m35 + b * b * m55;
      outErr(n, 0, 1) -= a * m35 + b * m55;
      outErr(n, 1, 1) += m55;
      outErr(n, 2, 2) += m44;
    }
  }

}  // namespace mkfit
