import flwr as fl

def main():
    # Start a Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",  # Adjust for your needs
        config={"num_rounds": 3},       # Specify the number of rounds
    )

if __name__ == "__main__":
    main()
