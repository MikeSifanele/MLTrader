using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using System.IO;
using Unity.MLAgents.Actuators;
using System.Globalization;
using Unity.MLAgents.Sensors;

public class MLHelper : Agent
{
    private float _accuracySum = 0;
    private int _epoch = 0;
    private readonly MLTrader _trader = new MLTrader();
    public override void OnEpisodeBegin()
    {
        Debug.Log("Episode started.");
        _trader.Reset();
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(_trader.GetObservation());
    }
    public override void OnActionReceived(ActionBuffers actions)
    {
        try
        {
            AddReward(_trader.GetReward(actions.DiscreteActions[0]));

            if (_trader.IsLastStep)
            {
                OnEndEpisode();
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"On Action Received: {ex.Message}");
        }        
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        int action = _trader.Target;
        var reward = _trader.GetReward(action);

        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;

        discreteActions[0] = action;

        if (reward == 1)
            Debug.Log($"Heuristically moved to Step no: {_trader.CurrentStepIndex}");
        else
            Debug.LogWarning($"Heuristically moved to Step no: {_trader.CurrentStepIndex}");

        if (_trader.IsLastStep)
        {
            OnEndEpisode();
        }
    }
    public void OnEndEpisode()
    {
        _epoch++;

        var reward = GetCumulativeReward();
        var maximumReward = _trader.MaximumRewards;

        var rewardString = reward.ToString("N", CultureInfo.CreateSpecificCulture("sv-SE"));
        var maximumRewardString = maximumReward.ToString("N", CultureInfo.CreateSpecificCulture("sv-SE"));

        _accuracySum += reward / maximumReward * 100;

        Debug.Log($"Episode ended: {_epoch}\nReward: {rewardString}/{maximumRewardString}\nAccuracy: {reward / maximumReward * 100:f1}%\nAverage Accuracy: {_accuracySum / _epoch:f1}%");

        _trader.Reset();

        EndEpisode();
    }
}
public enum SignalEnum
{
    Neutral = 0,
    FastValley = 1,
    SlowValley = 2,
    FastPeak = 3,
    SlowPeak = 4,
    Count
}
public struct Rates
{
    public string Time;
    public float Open;
    public float High;
    public float Low;
    public float Close;
    public SignalEnum Signal;
    public float FastEma;
    public float SlowEma;

    public Rates(string[] data)
    {
        Time = data[0];

        Open = float.Parse(data[1], CultureInfo.InvariantCulture.NumberFormat);
        High = float.Parse(data[2], CultureInfo.InvariantCulture.NumberFormat);
        Low = float.Parse(data[3], CultureInfo.InvariantCulture.NumberFormat);
        Close = float.Parse(data[4], CultureInfo.InvariantCulture.NumberFormat);

        Signal = (SignalEnum)int.Parse(data[5]);

        FastEma = float.Parse(data[6], CultureInfo.InvariantCulture.NumberFormat);
        SlowEma = float.Parse(data[7], CultureInfo.InvariantCulture.NumberFormat);
    }
    public float[] ToFloat()
    {
        return new float[] { FastEma, SlowEma, Open, High, Low, Close };
    }
}
public class MLTrader
{
    #region Private fields
    private Rates[] _rates;
    private readonly int _observationLength = 50;
    private int _index;
    /// <summary>
    /// Active or current non-neutral signal.
    /// </summary>
    private SignalEnum _currentSignal;
    #endregion
    #region Public properties
    public int CurrentStepIndex => _index - _observationLength;
    public bool IsLastStep => _index == MaximumRates - 1;
    public int MaximumRates => _rates.Length;
    public int MaximumRewards => MaximumRates - _observationLength;
    public SignalEnum CurrentSignal => _currentSignal;
    #endregion
    private static MLTrader _instance;
    public static MLTrader Instance => _instance ?? (_instance = new MLTrader());
    public MLTrader()
    {
        using (var streamReader = new StreamReader(Application.streamingAssetsPath + "/rates_rates.DAT"))
        {
            List<Rates> rates = new List<Rates>();

            _ = streamReader.ReadLine();

            while (!streamReader.EndOfStream)
            {
                rates.Add(new Rates(streamReader.ReadLine().Split(',')));
            }

            if (rates[0].Signal > 0)
                _currentSignal = rates[0].Signal;

            _rates = rates.ToArray();
        }

        Reset();
    }
    public float[] GetObservation()
    {
        List<float> observation = new List<float>();

        for (int i = _index - (_observationLength - 1); i <= _index; i++)
        {
            observation.AddRange(_rates[i].ToFloat());

            if (_rates[i].Signal != SignalEnum.Neutral)
                _currentSignal = _rates[i].Signal;
        }

        _index++;

        return observation.ToArray();
    }

    public float GetReward(int action)
    {
        return GetPoints(action) ?? 0f;
    }
    private float? GetPoints(int action, float? openPrice = null, float? closePrice = null)
    {
        openPrice ??= _rates[_index].Open;
        closePrice ??= _rates[_index].Close;

        return action == 0 ? (closePrice - openPrice) * 10 : (openPrice - closePrice) * 10;
    }
    public void Reset() => _index = _observationLength;
}
