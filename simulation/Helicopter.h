/*
 * Copyright 2015 Saminda Abeyruwan (saminda@cs.miami.edu)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Helicopter.h
 *
 *  Created on: May 12, 2013
 *      Author: sam
 */

#ifndef HELICOPTER_H_
#define HELICOPTER_H_

#include "RL.h"

// ============================================================================

class HeliVector
{
  public:
    double x, y, z;
    HeliVector() :
        x(0), y(0), z(0)
    {
    }

    HeliVector(const double& x, const double& y, const double& z) :
        x(x), y(y), z(z)
    {
    }

    void reset()
    {
      x = y = z = 0.0;
    }

};

class Quaternion
{
  public:
    double x, y, z, w;

    Quaternion() :
        x(0), y(0), z(0), w(0)
    {
    }
    Quaternion(const double& x, const double& y, const double& z, const double& w) :
        x(x), y(y), z(z), w(w)
    {
    }

    Quaternion(const HeliVector& v) :
        x(v.x), y(v.y), z(v.z), w(0)
    {
    }

    Quaternion conj() const
    {
      return Quaternion(-x, -y, -z, w);
    }

    HeliVector complex_part() const
    {
      return HeliVector(x, y, z);
    }

    Quaternion mult(const Quaternion& rq) const
    {
      return Quaternion(w * rq.x + x * rq.w + y * rq.z - z * rq.y,
          w * rq.y - x * rq.z + y * rq.w + z * rq.x, w * rq.z + x * rq.y - y * rq.x + z * rq.w,
          w * rq.w - x * rq.x - y * rq.y - z * rq.z);
    }

    void reset()
    {
      x = y = z = 0.0;
      w = 1.0;
    }
};

// Some transformations needed for this problem
class T
{
  public:
    static Quaternion to_quaternion(const HeliVector& v)
    {
      Quaternion quat;
      double rotation_angle = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      if (rotation_angle < 1e-4)
      { // avoid division by zero -- also: can use
        // simpler computation in this case, since for
        // small angles sin(x) = x is a good
        // approximation
        quat = Quaternion(v.x / 2.0f, v.y / 2.0f, v.z / 2.0f, 0.0f);
        quat.w = sqrt(1.0f - (quat.x * quat.x + quat.y * quat.y + quat.z * quat.z));
      }
      else
      {
        quat = Quaternion(sin(rotation_angle / 2.0f) * (v.x / rotation_angle),
            sin(rotation_angle / 2.0f) * (v.y / rotation_angle),
            sin(rotation_angle / 2.0f) * (v.z / rotation_angle), cos(rotation_angle / 2.0f));
      }
      return quat;
    }

    static HeliVector rotate(const HeliVector& v, const Quaternion& q)
    {
      return q.mult(Quaternion(v)).mult(q.conj()).complex_part();
    }

    static HeliVector express_in_quat_frame(const HeliVector& v, const Quaternion& q)
    {
      return rotate(v, q.conj());
    }
};

// ============================================================================
class HelicopterDynamics
{
  protected:
    /* some constants indexing into the helicopter's state */
    int ndot_idx; // north velocity
    int edot_idx; // east velocity
    int ddot_idx; // down velocity
    int n_idx; // north
    int e_idx; // east
    int d_idx; // down
    int p_idx; // angular rate around forward axis
    int q_idx; // angular rate around sideways (to the right)
               // axis
    int r_idx; // angular rate around vertical (downward) axis
    int qx_idx; // quaternion entries, x,y,z,w q = [ sin(theta/2)
                // * axis; cos(theta/2)]
    int qy_idx; // where axis = axis of rotation; theta is
                // amount of rotation around that axis
    int qz_idx; // [recall: any rotation can be represented by a
                // single rotation around some axis]
    int qw_idx;
    int state_size;

    int NUMOBS;
    // note: observation returned is not the state itself, but the "error state"
    // expressed in the helicopter's frame (which allows for a simpler mapping
    // from observation to inputs)
    // observation consists of:
    // u, v, w : velocities in helicopter frame
    // xerr, yerr, zerr: position error expressed in frame attached to helicopter
    // [xyz correspond to ned when helicopter is in "neutral" orientation, i.e.,
    // level and facing north]
    // p, q, r
    // qx, qy, qz

    vector<double> wind;

    // upper bounds on values state variables can take on (required by rl_glue to
    // be put into a string at environment initialization)
  public:
    double MaxVel; // m/s
    double MaxPos;
    double MaxRate;
    double MAX_QUAT;
    double MIN_QW_BEFORE_HITTING_TERMINAL_STATE;
    double MaxAction;
    double WIND_MAX;

    //
    vector<Range<double> > ObservationRanges;
    vector<Range<double> > ActionRanges;

  protected:
    // very crude helicopter model, okay around hover:
    double heli_model_u_drag;
    double heli_model_v_drag;
    double heli_model_w_drag;
    double heli_model_p_drag;
    double heli_model_q_drag;
    double heli_model_r_drag;
    double heli_model_u0_p;
    double heli_model_u1_q;
    double heli_model_u2_r;
    double heli_model_u3_w;
    double heli_model_tail_rotor_side_thrust;
    double DeltaT; // simulation time scale [time scale
                   // for
    // control ---
    // internally we integrate at 100Hz for simulating the
    // dynamics]

  public:
    HeliVector velocity;
    HeliVector position;
    HeliVector angularRate;
    Quaternion q;

  protected:
    vector<double> noise;
    vector<double> observation;

  public:
    HelicopterDynamics()
    {
      ndot_idx = 0; // north velocity
      edot_idx = 1; // east velocity
      ddot_idx = 2; // down velocity
      n_idx = 3; // north
      e_idx = 4; // east
      d_idx = 5; // down
      p_idx = 6; // angular rate around forward axis
      q_idx = 7; // angular rate around sideways (to the right)
                 // axis
      r_idx = 8; // angular rate around vertical (downward) axis
      qx_idx = 9; // quaternion entries, x,y,z,w q = [ sin(theta/2)
                  // * axis; cos(theta/2)]
      qy_idx = 10; // where axis = axis of rotation; theta is
                   // amount of rotation around that axis
      qz_idx = 11; // [recall: any rotation can be represented by a
                   // single rotation around some axis]
      qw_idx = 12;
      state_size = 13;

      NUMOBS = 12;
      wind.resize(2, 0);

      MaxVel = 5.0;
      MaxPos = 20.0;
      MaxRate = 2 * 3.1415 * 2;
      MAX_QUAT = 1.0;
      MIN_QW_BEFORE_HITTING_TERMINAL_STATE = cos(30.0 / 2.0 * M_PI / 180.0);
      MaxAction = 1.0;
      WIND_MAX = 5.0; //

      heli_model_u_drag = 0.18;
      heli_model_v_drag = 0.43;
      heli_model_w_drag = 0.49;
      heli_model_p_drag = 12.78;
      heli_model_q_drag = 10.12;
      heli_model_r_drag = 8.16;
      heli_model_u0_p = 33.04;
      heli_model_u1_q = -33.32;
      heli_model_u2_r = 70.54;
      heli_model_u3_w = -42.15;
      heli_model_tail_rotor_side_thrust = -0.54;
      DeltaT = .1;

      ObservationRanges.push_back(Range<double>(-MaxVel, MaxVel));
      ObservationRanges.push_back(Range<double>(-MaxVel, MaxVel));
      ObservationRanges.push_back(Range<double>(-MaxVel, MaxVel));
      ObservationRanges.push_back(Range<double>(-MaxPos, MaxPos));

      ObservationRanges.push_back(Range<double>(-MaxPos, MaxPos));
      ObservationRanges.push_back(Range<double>(-MaxPos, MaxPos));
      ObservationRanges.push_back(Range<double>(-MaxRate, MaxRate));
      ObservationRanges.push_back(Range<double>(-MaxRate, MaxRate));

      ObservationRanges.push_back(Range<double>(-MaxRate, MaxRate));
      ObservationRanges.push_back(Range<double>(-MAX_QUAT, MAX_QUAT));
      ObservationRanges.push_back(Range<double>(-MAX_QUAT, MAX_QUAT));
      ObservationRanges.push_back(Range<double>(-MAX_QUAT, MAX_QUAT));

      ActionRanges.push_back(Range<double>(-MaxAction, MaxAction));
      ActionRanges.push_back(Range<double>(-MaxAction, MaxAction));
      ActionRanges.push_back(Range<double>(-MaxAction, MaxAction));
      ActionRanges.push_back(Range<double>(-MaxAction, MaxAction));

      velocity = HeliVector(0.0, 0.0, 0.0);
      position = HeliVector(0.0, 0.0, 0.0);
      angularRate = HeliVector(0.0, 0.0, 0.0);
      q = Quaternion(0.0, 0.0, 0.0, 1.0);
      noise.resize(6, 0);

      observation.resize(NUMOBS, 0);

    }

    virtual ~HelicopterDynamics()
    {
    }

    void reset()
    {
      velocity.reset();
      position.reset();
      angularRate.reset();
      q.reset();
    }

    const vector<double>& getObservation()
    {
      // observation is the error state in the helicopter's coordinate system
      // (that way errors/observations can be mapped more directly to actions)
      HeliVector ned_error_in_heli_frame = T::express_in_quat_frame(position, q);
      HeliVector uvw = T::express_in_quat_frame(velocity, q);

      observation[0] = uvw.x;
      observation[1] = uvw.y;
      observation[2] = uvw.z;

      observation[n_idx] = ned_error_in_heli_frame.x;
      observation[e_idx] = ned_error_in_heli_frame.y;
      observation[d_idx] = ned_error_in_heli_frame.z;
      observation[p_idx] = angularRate.x;
      observation[q_idx] = angularRate.y;
      observation[r_idx] = angularRate.z;

      // the error quaternion gets negated, b/c we consider the rotation required
      // to bring the helicopter back to target in the helicopter's frame
      observation[qx_idx] = q.x;
      observation[qy_idx] = q.y;
      observation[qz_idx] = q.z;

      for (int i = 0; i < NUMOBS; i++)
        observation[i] = ObservationRanges[i].bound(observation[i]);
      return observation;
    }

    void step(Random<double>* random, const Action<double>* agentAction)
    {
      static double a[4];
      // saturate all the actions, b/c the actuators are limited:
      // [real helicopter's saturation is of course somewhat different, depends on
      // swash plate mixing etc ... ]
      for (int a_i = 0; a_i < 4; a_i++)
        a[a_i] = ActionRanges[a_i].bound(agentAction->getEntry(a_i));

      static double noise_mult = 2.0;
      static double noise_std[] = { 0.1941, 0.2975, 0.6058, 0.1508, 0.2492, 0.0734 }; // u,
                                                                                      // v,
                                                                                      // w,
                                                                                      // p,
                                                                                      // q,
                                                                                      // r
      double noise_memory = .8;
      // generate Gaussian random numbers

      for (int i = 0; i < 6; ++i)
        noise[i] = noise_memory * noise[i]
            + (1.0 - noise_memory) * random->nextNormalGaussian() * noise_std[i] * noise_mult;

      for (int t = 0; t < 10; ++t)
      {

        // Euler integration:

        // *** position ***
        position.x += DeltaT * velocity.x;
        position.y += DeltaT * velocity.y;
        position.z += DeltaT * velocity.z;

        // *** velocity ***
        HeliVector uvw = T::express_in_quat_frame(velocity, q);
        HeliVector wind_ned(wind[0], wind[1], 0.0);
        HeliVector wind_uvw = T::express_in_quat_frame(wind_ned, q);
        HeliVector uvw_force_from_heli_over_m(-heli_model_u_drag * (uvw.x + wind_uvw.x) + noise[0],
            -heli_model_v_drag * (uvw.y + wind_uvw.y) + heli_model_tail_rotor_side_thrust
                + noise[1], -heli_model_w_drag * uvw.z + heli_model_u3_w * a[3] + noise[2]);

        HeliVector ned_force_from_heli_over_m = T::rotate(uvw_force_from_heli_over_m, q);
        velocity.x += DeltaT * ned_force_from_heli_over_m.x;
        velocity.y += DeltaT * ned_force_from_heli_over_m.y;
        velocity.z += DeltaT * (ned_force_from_heli_over_m.z + 9.81);

        // *** orientation ***
        HeliVector axis_rotation(angularRate.x * DeltaT, angularRate.y * DeltaT,
            angularRate.z * DeltaT);
        Quaternion rot_quat = T::to_quaternion(axis_rotation);
        q = q.mult(rot_quat);

        // *** angular rate ***
        double p_dot = -heli_model_p_drag * angularRate.x + heli_model_u0_p * a[0] + noise[3];
        double q_dot = -heli_model_q_drag * angularRate.y + heli_model_u1_q * a[1] + noise[4];
        double r_dot = -heli_model_r_drag * angularRate.z + heli_model_u2_r * a[2] + noise[5];

        angularRate.x += DeltaT * p_dot;
        angularRate.y += DeltaT * q_dot;
        angularRate.z += DeltaT * r_dot;

      }
    }

    bool isCrashed() const
    {
      return (fabs(position.x) > MaxPos) || (fabs(position.y) > MaxPos)
          || (fabs(position.z) > MaxPos);
    }
};

template<class T>
class Helicopter: public RLProblem<T>
{
    typedef RLProblem<T> Base;
  protected:
    double episodeLength;
  public:
    double step_time;
    HelicopterDynamics heliDynamics;

  public:
    Helicopter(Random<double>* random, const int episodeLength = 6000) :
        RLProblem<T>(random, 12, 0, 1), episodeLength(episodeLength), step_time(0)
    {
      // Discrete actions are not setup for this problem.
      for (unsigned int i = 0; i < 4/*four values in the action*/; i++)
        Base::continuousActions->push_back(0, 0);
    }

    virtual ~Helicopter()
    {
    }

  private:
    double computeTerminalReward() const
    {
      double reward = -3.0f * heliDynamics.MaxPos * heliDynamics.MaxPos
          + -3.0f * heliDynamics.MaxRate * heliDynamics.MaxRate
          + -3.0f * heliDynamics.MaxVel * heliDynamics.MaxVel
          - (1.0f
              - heliDynamics.MIN_QW_BEFORE_HITTING_TERMINAL_STATE
                  * heliDynamics.MIN_QW_BEFORE_HITTING_TERMINAL_STATE);
      reward *= episodeLength - step_time;
      return reward;
    }

  public:
    void initialize()
    {
      heliDynamics.reset();
      step_time = 0;
    }

    void updateTRStep()
    {
      const vector<double>& observation = heliDynamics.getObservation();
      for (unsigned int i = 0; i < observation.size(); i++)
      {
        Base::output->observation_tp1->setEntry(i, observation[i]);
        Base::output->o_tp1->setEntry(i, observation[i]);
        // TODO: scaling?
      }
    }

    void step(const Action<T>* action)
    {
      heliDynamics.step(Base::random, action);
      ++step_time;
    }

    bool endOfEpisode() const
    {
      return heliDynamics.isCrashed() || step_time == episodeLength;
    }

    T r() const
    {
      if (heliDynamics.isCrashed())
        return computeTerminalReward();
      double reward = 0;
      reward -= heliDynamics.velocity.x * heliDynamics.velocity.x;
      reward -= heliDynamics.velocity.y * heliDynamics.velocity.y;
      reward -= heliDynamics.velocity.z * heliDynamics.velocity.z;
      reward -= heliDynamics.position.x * heliDynamics.position.x;
      reward -= heliDynamics.position.y * heliDynamics.position.y;
      reward -= heliDynamics.position.z * heliDynamics.position.z;
      reward -= heliDynamics.angularRate.x * heliDynamics.angularRate.x;
      reward -= heliDynamics.angularRate.y * heliDynamics.angularRate.y;
      reward -= heliDynamics.angularRate.z * heliDynamics.angularRate.z;
      reward -= heliDynamics.q.x * heliDynamics.q.x;
      reward -= heliDynamics.q.y * heliDynamics.q.y;
      reward -= heliDynamics.q.z * heliDynamics.q.z;
      return reward;
    }

    T z() const
    {
      return 0;
    }

};

#endif /* HELICOPTER_H_ */
