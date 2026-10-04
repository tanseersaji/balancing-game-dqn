using System.Net.Sockets;
using System.Text;
using System;
using TMPro;
using UnityEngine;
using UnityEngine.InputSystem;

public class BallController : MonoBehaviour
{

    public TMP_Text scoreLabel;
    public Rigidbody2D ball;

    public Collider2D[] walls;

    private Rigidbody2D platform;
    private float score = 0f;
    private static float rotationSpeed = 1000f;

    private TcpClient client;
    private NetworkStream stream;
    private byte[] buffer = new byte[16];

    private bool gameOverFlag = false;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;

        Time.timeScale = 1f;

        platform = GetComponent<Rigidbody2D>();

        try
        {
            client = new TcpClient("127.0.0.1", 5555);
            stream = client.GetStream();

        }
        catch (Exception e)
        {
            Debug.LogError($"Cannot connect to Brain - {e}");
        }
    }

    void Update()
    {
        score += Time.deltaTime;

        if (scoreLabel != null) scoreLabel.text = ((int)score).ToString();
    }

    void SendBallState()
    {
        float distanceToWall = IsGameOver();

        string stateMessage = $"{ball.position.x},{ball.position.y},{ball.linearVelocity.x},{ball.linearVelocity.y},{platform.rotation},{distanceToWall},{(gameOverFlag ? 1 : 0)}\n";
        byte[] stateBytes = Encoding.UTF8.GetBytes(stateMessage);
        stream.Write(stateBytes, 0, stateBytes.Length);
    }

    void FixedUpdate()
    {
        int action = 0;
        if (stream != null && stream.CanWrite)
        {
            SendBallState();
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string actionStr = Encoding.UTF8.GetString(buffer, 0, bytesRead).Trim();
            int.TryParse(actionStr, out action);
        }

        if (gameOverFlag)
        {
            ResetState();
        }
        else
        {
            if (action == 1) {platform.AddTorque(-rotationSpeed * Time.fixedDeltaTime); }
            if (action == 2) {platform.AddTorque(rotationSpeed * Time.fixedDeltaTime); }
        }
    }

    void OnApplicationQuit()
    {
        stream?.Close();
        client?.Close();
    }

    void ResetState()
    {
        if (ball != null)
        {
            float randomX = UnityEngine.Random.Range(-0.5f, 0.5f);
            ball.position = new Vector2(randomX, platform.position.y + 2f);
            ball.linearVelocity = Vector2.zero;
            ball.angularVelocity = 0f;
        }

        float randomRotation = UnityEngine.Random.Range(-10f, 10f);
        platform.rotation = randomRotation;
        platform.angularVelocity = 0f;

        score = 0;
        if (scoreLabel != null) scoreLabel.text = ((int)score).ToString();

        gameOverFlag = false;
    }

    float IsGameOver()
    {
        float tempDistance = 1000f;
        Collider2D ballCollider = ball.GetComponent<Collider2D>();
        for (int i = 0; i < walls.Length; i++)
        {
            Collider2D wall = walls[i];
            if (tempDistance > wall.Distance(ballCollider).distance) tempDistance = wall.Distance(ballCollider).distance;

            if (walls[i].bounds.Intersects(ballCollider.bounds))
            {
                gameOverFlag = true;
                break;
            }
        }

        return tempDistance;
    }
}
