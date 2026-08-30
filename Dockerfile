FROM rust:1.75 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release -p thompson-sim && cargo build --release -p control-plane

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/thompson-sim /usr/local/bin/thompson-sim
EXPOSE 8080
ENTRYPOINT ["thompson-sim"]
CMD ["--help"]
