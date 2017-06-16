class Data::Api::OilFutureService
  class Error < RuntimeError; end
  class IntegrationError < Error; end

  include TimeHelper
  include ErrorHelper

  RETRY_COUNT = 3

  def self.call_service
    new.call_service
  end

  def call_service(retry_count: RETRY_COUNT)
    call()
  rescue => e
    sleep(TimeHelper.exponential_drop_off(base: ((RETRY_COUNT - retry_count) + 2)))
    retry_count -= 1
    retry if retry_count > 0
    ErrorHelper.mail_error('Oil future service error: ', e.message)
  end

  def call
    prev_date = Date.today - 1
    start_date = OilFuture.order(date: :desc).first.date + 1

    return if start_date > prev_date

    puts ("call oil future for: #{start_date.to_s} - #{prev_date.to_s}")

    uri = URI("https://www.quandl.com/api/v3/datasets/CHRIS/ICE_B1.json?api_key=#{Rails.application.secrets.oil_price_key}&start_date=#{start_date.to_s}&end_date=#{prev_date.to_s}")
    service_request = Net::HTTP.get_response(uri)

    if service_request.response.is_a?(Net::HTTPSuccess)
      data = JSON.parse(service_request.body).to_h['dataset']['data']
      save_oil_future(data) if data.any?
      puts ('SUCCESS!')
    else
      raise IntegrationError.new("Unable to successfully call oil future api, error: #{service_request.body.to_s}")
    end
  end

  def save_oil_future(data)
    data.each do |item|
      next if OilFuture.find_by_date(item[0]).present?

      OilFuture.create(currency: 'USD',
                       date: item[0],
                       settle: item[4],
                       wave: item[6],
                       volume: item[7],
                       efp_volume: item[9],
                       efs_volume: item[10],
                       block_volume: item[11],
                       source: 'https://www.quandl.com/data/CHRIS/ICE_B1-Brent-Crude-Futures-Continuous-Contract-1-B1-Front-Month')
    end
  end
end