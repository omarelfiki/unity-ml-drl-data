using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MoveToTargetAgent : Agent
{
    public Transform target;
    public float moveSpeed = 5f;
    public float arenaSize = 5f;
    Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset agent velocity
        rb.linearVelocity = Vector3.zero;

        // Randomize agent & target positions inside a square arena
        transform.localPosition = new Vector3(
            Random.Range(-arenaSize, arenaSize), 0.5f, Random.Range(-arenaSize, arenaSize)
        );

        target.localPosition = new Vector3(
            Random.Range(-arenaSize, arenaSize), 0.5f, Random.Range(-arenaSize, arenaSize)
        );
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Observe relative direction to target, agent velocity (x,z)
        Vector3 toTarget = (target.localPosition - transform.localPosition);
        sensor.AddObservation(toTarget.x);
        sensor.AddObservation(toTarget.z);
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.z);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        var act = actions.ContinuousActions;
        Vector3 force = new Vector3(act[0], 0f, act[1]) * moveSpeed;
        rb.AddForce(force, ForceMode.VelocityChange);

        // Small step penalty to encourage efficiency
        AddReward(-0.0005f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxis("Horizontal");
        ca[1] = Input.GetAxis("Vertical");
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("Target"))
        {
            AddReward(+1.0f);
            EndEpisode();
        }
        else if (other.gameObject.CompareTag("Wall"))
        {
            AddReward(-0.5f);
            EndEpisode();
        }
    }
}