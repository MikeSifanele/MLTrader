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
        var reward = _trader.GetReward(_trader.Target);

        AddReward(reward);

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
    public Signal Signal;

    public Rates(string[] data)
    {
        Time = data[0];

        Open = float.Parse(data[1], CultureInfo.InvariantCulture.NumberFormat);
        High = float.Parse(data[2], CultureInfo.InvariantCulture.NumberFormat);
        Low = float.Parse(data[3], CultureInfo.InvariantCulture.NumberFormat);
        Close = float.Parse(data[4], CultureInfo.InvariantCulture.NumberFormat);

        Signal = new Signal(data);
    }
    public float[] ToFloat()
    {
        return new float[] { Open, High, Low, Close };
    }
}
public class Signal
{
    public SignalEnum Value;
    public Signal(string[] data)
    {
        if (data[5] != "0")
            Value = SignalEnum.FastValley;
        else if (data[6] != "0")
            Value = SignalEnum.SlowValley;
        else if (data[7] != "0")
            Value = SignalEnum.FastPeak;
        else if (data[8] != "0")
            Value = SignalEnum.SlowPeak;
        else
            Value = SignalEnum.Neutral;
    }
}
public class MLTrader
{
    #region Private fields
    private Rates[] _rates;
    private readonly int _observationLength = 50;
    private int _index;
    #endregion
    #region Public properties
    public int CurrentStepIndex => _index - _observationLength;
    public bool IsLastStep => _index == MaximumRates;
    public int MaximumRates => _rates.Length;
    public int MaximumRewards => MaximumRates - _observationLength;
    public int Target => (int)_rates[_index].Signal.Value;
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

            _rates = rates.ToArray();
        }

        Reset();
    }
    public float[] GetObservation()
    {
        List<float> observation = new List<float>();

        for (int i = _index+1; i > _index - _observationLength; i--)
            observation.AddRange(_rates[i].ToFloat());

        _index++;

        return observation.ToArray();
    }

    public int GetReward(int action)
    {
        SignalEnum signal = _rates[_index].Signal.Value;

        switch ((SignalEnum)action)
        {
            case SignalEnum.FastPeak:
                if(signal == SignalEnum.Neutral)
                    return -100;
                if(signal == SignalEnum.SlowPeak)
                    return -1;
                if (signal == SignalEnum.FastValley)
                    return -10;
                else if (signal == SignalEnum.SlowValley)
                    return -100;
                break;
            case SignalEnum.FastValley:
                if (signal == SignalEnum.Neutral)
                    return -100;
                if (signal == SignalEnum.SlowValley)
                    return -1;
                if (signal == SignalEnum.FastPeak)
                    return -10;
                else if (signal == SignalEnum.SlowPeak)
                    return -100;
                break;
            case SignalEnum.SlowPeak:
                if (signal == SignalEnum.Neutral)
                    return -100;
                if (signal == SignalEnum.FastPeak)
                    return -1;
                if (signal == SignalEnum.FastValley)
                    return -10;
                else if (signal == SignalEnum.SlowValley)
                    return -100;
                break;
            case SignalEnum.SlowValley:
                if (signal == SignalEnum.Neutral)
                    return -100;
                if (signal == SignalEnum.FastValley)
                    return -1;
                if (signal == SignalEnum.FastPeak)
                    return -10;
                else if (signal == SignalEnum.SlowPeak)
                    return -100;
                break;
            case SignalEnum.Neutral:
                if (signal == SignalEnum.FastPeak)
                    return -10;
                else if (signal == SignalEnum.SlowPeak)
                    return -100;
                else if (signal == SignalEnum.FastValley)
                    return -10;
                else if (signal == SignalEnum.SlowValley)
                    return -100;
                break;
        }

        return 1;
    }

    public void Reset() => _index = _observationLength;
}
